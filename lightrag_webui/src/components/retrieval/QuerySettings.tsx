import { useCallback, useMemo } from 'react'
import { QueryMode, QueryRequest } from '@/api/lightrag'
// Removed unused import for Text component
import Checkbox from '@/components/ui/Checkbox'
import Input from '@/components/ui/Input'
import UserPromptInputWithHistory from '@/components/ui/UserPromptInputWithHistory'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'
import { useSettingsStore } from '@/stores/settings'
import { useTranslation } from 'react-i18next'
import { RotateCcw } from 'lucide-react'

export default function QuerySettings() {
  const { t } = useTranslation()
  const querySettings = useSettingsStore((state) => state.querySettings)
  const userPromptHistory = useSettingsStore((state) => state.userPromptHistory)

  const handleChange = useCallback((key: keyof QueryRequest, value: any) => {
    useSettingsStore.getState().updateQuerySettings({ [key]: value })
  }, [])

  const handleSelectFromHistory = useCallback((prompt: string) => {
    handleChange('user_prompt', prompt)
  }, [handleChange])

  const handleDeleteFromHistory = useCallback((index: number) => {
    const newHistory = [...userPromptHistory]
    newHistory.splice(index, 1)
    useSettingsStore.getState().setUserPromptHistory(newHistory)
  }, [userPromptHistory])

  // Default values for reset functionality
  const defaultValues = useMemo(() => ({
    mode: 'naive' as QueryMode,
    top_k: 40,
    chunk_top_k: 20,
    max_entity_tokens: 6000,
    max_relation_tokens: 8000,
    max_total_tokens: 30000
  }), [])

  const handleReset = useCallback((key: keyof typeof defaultValues) => {
    handleChange(key, defaultValues[key])
  }, [handleChange, defaultValues])

  // Reset button component
  const ResetButton = ({ onClick, title }: { onClick: () => void; title: string }) => (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={onClick}
            className="mr-1 p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            title={title}
          >
            <RotateCcw className="h-3 w-3 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200" />
          </button>
        </TooltipTrigger>
        <TooltipContent side="left">
          <p>{title}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )

  return (
    <Card className="flex shrink-0 flex-col w-[280px] border border-slate-200/80 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-900/30 backdrop-blur-md shadow-[0_4px_20px_-2px_rgba(0,0,0,0.04)] dark:shadow-none">
      <CardHeader className="px-4 pt-4 pb-3 border-b border-slate-200/80 dark:border-slate-800/60 bg-slate-100/50 dark:bg-slate-900/40 rounded-t-xl">
        <CardTitle className="text-sm font-bold text-primary">{t('retrievePanel.querySettings.parametersTitle')}</CardTitle>
        <CardDescription className="sr-only">{t('retrievePanel.querySettings.parametersDescription')}</CardDescription>
      </CardHeader>
      <CardContent className="m-0 flex grow flex-col p-3 text-xs overflow-hidden">
        <div className="relative size-full">
          <div className="absolute inset-0 flex flex-col gap-3.5 overflow-auto pr-1 select-none">
            
            {/* Group 1: Chế độ truy vấn (User Prompt & Query Mode) */}
            <div className="p-3 bg-white dark:bg-slate-950/75 border border-slate-200/60 dark:border-slate-800/40 rounded-xl space-y-3 shrink-0 shadow-sm hover:shadow-md transition-all duration-300">
              <div className="text-[10px] font-bold uppercase tracking-wider text-primary">
                Chế độ truy vấn
              </div>
              
              {/* User Prompt */}
              <div className="space-y-1">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="user_prompt" className="ml-0.5 font-medium text-muted-foreground cursor-help">
                        {t('retrievePanel.querySettings.userPrompt')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.userPromptTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <UserPromptInputWithHistory
                  id="user_prompt"
                  value={querySettings.user_prompt || ''}
                  onChange={(value) => handleChange('user_prompt', value)}
                  onSelectFromHistory={handleSelectFromHistory}
                  onDeleteFromHistory={handleDeleteFromHistory}
                  history={userPromptHistory}
                  placeholder={t('retrievePanel.querySettings.userPromptPlaceholder')}
                  className="h-9 w-full bg-background border-slate-300/80 dark:border-slate-700/80 focus-visible:border-primary"
                />
              </div>

              {/* Query Mode */}
              <div className="space-y-1">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="query_mode_select" className="ml-0.5 font-medium text-muted-foreground cursor-help">
                        {t('retrievePanel.querySettings.queryMode')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.queryModeTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-1.5">
                  <Select
                    value={querySettings.mode}
                    onValueChange={(v) => handleChange('mode', v as QueryMode)}
                  >
                    <SelectTrigger
                      id="query_mode_select"
                      className="hover:bg-primary/5 h-9 cursor-pointer focus:ring-0 focus:ring-offset-0 focus:outline-0 flex-1 text-left bg-background border-slate-300/80 dark:border-slate-700/80 [&>span]:break-all [&>span]:line-clamp-1"
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectItem value="naive">{t('retrievePanel.querySettings.queryModeOptions.naive')}</SelectItem>
                        <SelectItem value="hybrid">{t('retrievePanel.querySettings.queryModeOptions.hybrid')}</SelectItem>
                        <SelectItem value="stat">{t('retrievePanel.querySettings.queryModeOptions.stat')}</SelectItem>
                        <SelectItem value="mix">{t('retrievePanel.querySettings.queryModeOptions.mix')}</SelectItem>
                        <SelectItem value="graph">{t('retrievePanel.querySettings.queryModeOptions.graph')}</SelectItem>
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                  <ResetButton
                    onClick={() => handleReset('mode')}
                    title="Reset to default (Mix)"
                  />
                </div>
              </div>
            </div>

            {/* Group 2: Tham số thu hồi (Chunk Top K & Max Total Tokens) */}
            <div className="p-3 bg-white dark:bg-slate-950/75 border border-slate-200/60 dark:border-slate-800/40 rounded-xl space-y-3 shrink-0 shadow-sm hover:shadow-md transition-all duration-300">
              <div className="text-[10px] font-bold uppercase tracking-wider text-primary">
                Tham số thu hồi
              </div>

              {/* Chunk Top K */}
              <div className="space-y-1">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="chunk_top_k" className="ml-0.5 font-medium text-muted-foreground cursor-help">
                        {t('retrievePanel.querySettings.chunkTopK')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.chunkTopKTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-1.5">
                  <Input
                    id="chunk_top_k"
                    type="number"
                    value={querySettings.chunk_top_k ?? ''}
                    onChange={(e) => {
                      const value = e.target.value
                      handleChange('chunk_top_k', value === '' ? '' : parseInt(value) || 0)
                    }}
                    onBlur={(e) => {
                      const value = e.target.value
                      if (value === '' || isNaN(parseInt(value))) {
                        handleChange('chunk_top_k', 20)
                      }
                    }}
                    min={1}
                    placeholder={t('retrievePanel.querySettings.chunkTopKPlaceholder')}
                    className="h-9 flex-1 bg-background border-slate-300/80 dark:border-slate-700/80 pr-2 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
                  />
                  <ResetButton
                    onClick={() => handleReset('chunk_top_k')}
                    title="Reset to default"
                  />
                </div>
              </div>

              {/* Max Total Tokens */}
              <div className="space-y-1">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="max_total_tokens" className="ml-0.5 font-medium text-muted-foreground cursor-help">
                        {t('retrievePanel.querySettings.maxTotalTokens')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.maxTotalTokensTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-1.5">
                  <Input
                    id="max_total_tokens"
                    type="number"
                    value={querySettings.max_total_tokens ?? ''}
                    onChange={(e) => {
                      const value = e.target.value
                      handleChange('max_total_tokens', value === '' ? '' : parseInt(value) || 0)
                    }}
                    onBlur={(e) => {
                      const value = e.target.value
                      if (value === '' || isNaN(parseInt(value))) {
                        handleChange('max_total_tokens', 30000)
                      }
                    }}
                    min={1}
                    placeholder={t('retrievePanel.querySettings.maxTotalTokensPlaceholder')}
                    className="h-9 flex-1 bg-background border-slate-300/80 dark:border-slate-700/80 pr-2 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
                  />
                  <ResetButton
                    onClick={() => handleReset('max_total_tokens')}
                    title="Reset to default"
                  />
                </div>
              </div>
            </div>

            {/* Group 3: Cấu hình phản hồi (Toggles) */}
            <div className="p-3 bg-white dark:bg-slate-950/75 border border-slate-200/60 dark:border-slate-800/40 rounded-xl space-y-3 shrink-0 shadow-sm hover:shadow-md transition-all duration-300">
              <div className="text-[10px] font-bold uppercase tracking-wider text-primary mb-1">
                Cấu hình phản hồi
              </div>

              <div className="flex items-center justify-between gap-2 py-0.5">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="enable_rerank" className="cursor-help text-muted-foreground font-medium">
                        {t('retrievePanel.querySettings.enableRerank')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.enableRerankTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <Checkbox
                  className="cursor-pointer"
                  id="enable_rerank"
                  checked={querySettings.enable_rerank}
                  onCheckedChange={(checked) => handleChange('enable_rerank', checked)}
                />
              </div>

              <div className="flex items-center justify-between gap-2 py-0.5">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="only_need_context" className="cursor-help text-muted-foreground font-medium">
                        {t('retrievePanel.querySettings.onlyNeedContext')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.onlyNeedContextTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <Checkbox
                  className="cursor-pointer"
                  id="only_need_context"
                  checked={querySettings.only_need_context}
                  onCheckedChange={(checked) => {
                    handleChange('only_need_context', checked)
                    if (checked) {
                      handleChange('only_need_prompt', false)
                    }
                  }}
                />
              </div>

              <div className="flex items-center justify-between gap-2 py-0.5">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="only_need_prompt" className="cursor-help text-muted-foreground font-medium">
                        {t('retrievePanel.querySettings.onlyNeedPrompt')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.onlyNeedPromptTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <Checkbox
                  className="cursor-pointer"
                  id="only_need_prompt"
                  checked={querySettings.only_need_prompt}
                  onCheckedChange={(checked) => {
                    handleChange('only_need_prompt', checked)
                    if (checked) {
                      handleChange('only_need_context', false)
                    }
                  }}
                />
              </div>

              <div className="flex items-center justify-between gap-2 py-0.5">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="stream" className="cursor-help text-muted-foreground font-medium">
                        {t('retrievePanel.querySettings.streamResponse')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.streamResponseTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <Checkbox
                  className="cursor-pointer"
                  id="stream"
                  checked={querySettings.stream}
                  onCheckedChange={(checked) => handleChange('stream', checked)}
                />
              </div>
            </div>

          </div>
        </div>
      </CardContent>
    </Card>
  )
}
