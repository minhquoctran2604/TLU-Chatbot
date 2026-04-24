import logging
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
)
from docling_core.transforms.serializer.base import (
    BasePictureSerializer,
    BaseDocSerializer,
    BaseSerializerProvider,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import (
    DoclingDocument,
    PictureItem,
)
from lightrag.lightrag import LightRAG

_HARRIER_MODEL: SentenceTransformer | None = None
_HARRIER_DIM: int | None = None


def _get_harrier_model() -> SentenceTransformer:
    global _HARRIER_MODEL, _HARRIER_DIM
    if _HARRIER_MODEL is None:
        _HARRIER_MODEL = SentenceTransformer("microsoft/harrier-oss-v1-270m")
        _HARRIER_DIM = _HARRIER_MODEL.get_sentence_embedding_dimension()
    return _HARRIER_MODEL


def _get_harrier_dim() -> int:
    global _HARRIER_DIM
    if _HARRIER_DIM is None:
        _get_harrier_model()
    return _HARRIER_DIM


async def _noop_llm(*args, **kwargs) -> str:
    return ""


async def harrier_embed(texts: List[str], **kwargs) -> np.ndarray:
    model = _get_harrier_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.array(embeddings)


logger = logging.getLogger(__name__)

MIN_IMG_WIDTH = 50
MIN_IMG_HEIGHT = 50


class MarkerPictureSerializer(BasePictureSerializer):
    def __init__(self, doc_name: str, images_dir: Path, decorative_hashes: set = None):
        self._counter = 0
        self._doc_name = doc_name
        self._images_dir = images_dir
        self._skipped = 0
        self._skipped_decorative = 0
        self._decorative_hashes = decorative_hashes or set()
        self.image_files = {}  # {marker_string: filename}

    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        if item.image is not None:
            pil_img = item.image.pil_image
            if pil_img is not None:
                if pil_img.width < MIN_IMG_WIDTH or pil_img.height < MIN_IMG_HEIGHT:
                    self._skipped += 1
                    return create_ser_result(text="")

                # Filter decorative (logos/templates) detected by PDF forensics
                if self._decorative_hashes:
                    from lightrag.pdf_forensics import is_decorative
                    if is_decorative(pil_img, self._decorative_hashes):
                        self._skipped_decorative += 1
                        return create_ser_result(text="")

                idx = self._counter
                self._counter += 1
                marker = f"[IMG_{self._doc_name}_{idx}]"

                if pil_img.width > 800:
                    ratio = 800 / pil_img.width
                    pil_img = pil_img.resize((800, int(pil_img.height * ratio)))

                filename = f"{self._doc_name}_img{idx}.png"
                filepath = self._images_dir / filename
                try:
                    pil_img.save(filepath, "PNG")
                    self.image_files[marker] = filename
                except Exception as e:
                    logger.error(f"Save {marker} failed: {e}")

                return create_ser_result(text=marker, span_source=item)

        return create_ser_result(text="")

class MarkerSerializerProvider(BaseSerializerProvider):
    def __init__(self, doc_name: str, images_dir: Path, decorative_hashes: set = None):
        self._doc_name = doc_name
        self._images_dir = images_dir
        self._decorative_hashes = decorative_hashes or set()
        self.picture_serializer = None

    def get_serializer(self, doc: DoclingDocument) -> BaseDocSerializer:
        self.picture_serializer = MarkerPictureSerializer(
            self._doc_name, self._images_dir, self._decorative_hashes
        )
        return ChunkingDocSerializer(
            doc=doc,
            picture_serializer=self.picture_serializer,
        )

def sanitize_text(text: str) -> str:
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    text = re.sub(r"[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]", "", text)
    return text.strip()


def compute_chunk_id(text: str) -> str:
    return "chunk-" + hashlib.md5(sanitize_text(text).encode("utf-8")).hexdigest()


def safe_doc_name(rel_path: str) -> str:
    return Path(rel_path).with_suffix("").as_posix().replace("/", "_")


async def notebook_ingest_pdf(rag: LightRAG, pdf_path: Path) -> Dict[str, Any]:
    """Replicate the notebook ingest flow for a single uploaded PDF."""

    pdf_path = Path(pdf_path)
    working_dir = Path(rag.working_dir)

    # Pre-scan PDF for decorative images (logos, templates) via XObject reuse
    try:
        from lightrag.pdf_forensics import get_decorative_image_hashes
        decorative_hashes = get_decorative_image_hashes(str(pdf_path))
    except Exception as e:
        logger.warning(f"PDF forensics failed: {e}. Continuing without filter.")
        decorative_hashes = set()

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.do_ocr = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )

    result = converter.convert(str(pdf_path))
    doc = result.document

    md_path = pdf_path.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save_as_markdown(md_path, image_mode=ImageRefMode.REFERENCED)
    markdown = md_path.read_text(encoding="utf-8")

    doc_name = safe_doc_name(pdf_path.name)
    images_dir = working_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    provider = MarkerSerializerProvider(doc_name, images_dir, decorative_hashes)
    chunker = HybridChunker(
        max_tokens=512,
        merge_peers=True,
        repeat_table_header=True,
        serializer_provider=provider,
    )

    doc_chunks = list(chunker.chunk(dl_doc=doc))
    chunk_texts: List[str] = []
    image_mapping: Dict[str, Dict[str, str]] = {}

    for chunk in doc_chunks:
        chunk_text = chunker.contextualize(chunk)
        chunk_texts.append(chunk_text)
        chunk_id = compute_chunk_id(chunk_text)

        saved_images = provider.picture_serializer.image_files if provider.picture_serializer else {}
        markers_found = re.findall(r"\[IMG_[^\]]+\]", chunk_text)
        if markers_found:
            chunk_image_map = {
                marker: saved_images[marker]
                for marker in markers_found
                if marker in saved_images
            }
            if chunk_image_map:
                image_mapping[chunk_id] = chunk_image_map

    mapping_path = working_dir / "image_mapping.json"
    merged_image_mapping: Dict[str, Dict[str, str]] = {}
    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                existing_mapping = json.load(f)
                if isinstance(existing_mapping, dict):
                    merged_image_mapping.update(existing_mapping)
        except Exception as e:
            logger.warning(f"Failed to load existing image mapping: {e}")

    merged_image_mapping.update(image_mapping)

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(merged_image_mapping, f, ensure_ascii=False, indent=2)

    current_embedding = getattr(rag, "embedding_func", None)
    if current_embedding is None:
        raise RuntimeError(
            "Runtime LightRAG instance must provide embedding_func before notebook_ingest_pdf runs."
        )

    await rag.initialize_storages()
    await rag.ainsert_custom_chunks(
        full_text=markdown,
        text_chunks=chunk_texts,
        file_path=pdf_path.name,
    )

    skipped = provider.picture_serializer._skipped if provider.picture_serializer else 0
    skipped_decorative = provider.picture_serializer._skipped_decorative if provider.picture_serializer else 0
    saved_count = len(provider.picture_serializer.image_files) if provider.picture_serializer else 0

    logger.info(
        f"Image stats: saved={saved_count}, skipped_small={skipped}, "
        f"skipped_decorative={skipped_decorative}, blacklist_size={len(decorative_hashes)}"
    )

    return {
        "status": "success",
        "chunks": len(chunk_texts),
        "images": saved_count,
        "skipped": skipped,
        "mapping_path": str(mapping_path),
        "doc_name": doc_name,
    }
