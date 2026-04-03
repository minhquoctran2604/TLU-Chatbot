"""Text2Cypher module: convert natural language queries to Cypher for Neo4j.

Schema TLU:
  Nodes: Document, Person, Topic, Major, Degree
  Relationships: WROTE, ADVISED, HAS_TOPIC, BELONGS_TO_MAJOR, HAS_DEGREE

  Document props: id, type, raw_type, title, abstract, year, language, publisher, uri, oai_id, ddc, date
  Person props: name
  Topic props: name
  Major props: name
  Degree props: name
"""

import json
import logging
import os
from typing import Any, Optional

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

# Schema whitelist for validation
VALID_NODE_LABELS = {"Document", "Person", "Topic", "Major", "Degree"}
VALID_REL_TYPES = {"WROTE", "ADVISED", "HAS_TOPIC", "BELONGS_TO_MAJOR", "HAS_DEGREE"}
BLOCKED_OPERATIONS = {"CREATE", "DELETE", "SET", "REMOVE", "MERGE", "DROP", "DETACH"}

SCHEMA_DESCRIPTION = """
Neo4j Schema (TLU Digital Library):

Nodes:
- Document: id, type, raw_type, title, abstract, year, language, publisher, uri, oai_id, ddc, date
- Person: name
- Topic: name
- Major: name
- Degree: name

Relationships:
- (Person)-[:WROTE]->(Document)
- (Person)-[:ADVISED]->(Document)
- (Document)-[:HAS_TOPIC]->(Topic)
- (Document)-[:BELONGS_TO_MAJOR]->(Major)
- (Document)-[:HAS_DEGREE]->(Degree)

Notes:
- type is one of: Thesis, Article, Lecture, Publication, Standard, Journal, Other, Project, Report
- year is STRING (use d.year = "2020" not d.year = 2020)
- All text fields are in Vietnamese
""".strip()

CYPHER_SYSTEM_PROMPT = """You are a Cypher query generator for a Neo4j database.

{schema}

Rules:
1. ONLY use node labels and relationship types from the schema above
2. ONLY generate READ queries (MATCH/RETURN). Never write CREATE/DELETE/SET/MERGE
3. Use CONTAINS for partial text matching on Vietnamese text
4. Use toLower() for case-insensitive matching
5. Always LIMIT results to 20 or less
6. Return meaningful fields, not just counts (unless specifically asked)
7. Respond with ONLY the Cypher query, no explanation."""

CYPHER_USER_PROMPT = "{question}"


class CypherValidator:
    """Validate Cypher queries against schema whitelist."""

    @staticmethod
    def validate(cypher: str) -> tuple[bool, str]:
        """Return (is_valid, error_message)."""
        upper = cypher.upper()

        # Block write operations
        for op in BLOCKED_OPERATIONS:
            if op in upper and "CONSTRAINT" not in upper:
                return False, f"Blocked operation: {op}"

        return True, ""


class Neo4jExecutor:
    """Execute Cypher queries on Neo4j."""

    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()

        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME") or os.getenv("USERNAME_NEO4J")
        password = os.getenv("NEO4J_PASSWORD") or os.getenv("PASSWORD_NEO4J")

        if not all([uri, user, password]):
            logger.warning("Neo4j credentials not configured, Text2Cypher disabled")
            self.driver = None
            return

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.info("Neo4j connected for Text2Cypher")
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}")
            self.driver = None

    def execute(self, cypher: str) -> Optional[list[dict]]:
        """Execute Cypher query, return list of records or None on failure."""
        if not self.driver:
            return None

        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                records = [dict(record) for record in result]
                logger.info(f"Cypher returned {len(records)} records")
                return records
        except Exception as e:
            logger.warning(f"Cypher execution failed: {e}")
            return None

    def close(self):
        if self.driver:
            self.driver.close()


async def generate_cypher(query: str, llm_func, **kwargs) -> Optional[str]:
    """Use LLM to generate Cypher from natural language query."""
    system_prompt = CYPHER_SYSTEM_PROMPT.format(schema=SCHEMA_DESCRIPTION)
    user_prompt = CYPHER_USER_PROMPT.format(question=query)

    try:
        response = await llm_func(user_prompt, system_prompt=system_prompt, **kwargs)
        if not response:
            return None

        # Clean up response
        cypher = response.strip()
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()

        # Validate
        is_valid, error = CypherValidator.validate(cypher)
        if not is_valid:
            logger.warning(f"Cypher validation failed: {error}")
            return None

        logger.info(f"Generated Cypher: {cypher}")
        return cypher

    except Exception as e:
        logger.warning(f"Cypher generation failed: {e}")
        return None


def format_cypher_results(records: list[dict]) -> str:
    """Format Neo4j results as readable text for LLM context."""
    if not records:
        return ""

    lines = ["Dữ liệu từ cơ sở dữ liệu đồ thị (Neo4j):"]
    for i, record in enumerate(records[:20], 1):
        parts = []
        for key, value in record.items():
            if value is not None:
                parts.append(f"{key}: {value}")
        lines.append(f"  {i}. {', '.join(parts)}")

    return "\n".join(lines)
