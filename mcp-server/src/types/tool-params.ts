/**
 * MCP工具参数类型定义
 */

// 搜索代码示例参数
export interface SearchCodeExamplesParams {
  query: string;
  language?: 'python' | 'typescript' | 'javascript' | 'java' | 'go' | 'cpp';
  framework?: string;
  team?: string;
  project?: string;
  limit?: number;
}

// 获取设计文档参数
export interface GetDesignDocsParams {
  query: string;
  doc_type?: 'api_design' | 'architecture' | 'database_design' | 'business_logic';
  component?: string;
  team?: string;
  project?: string;
}

// 获取编码规范参数
export interface GetCodingStandardsParams {
  language: 'python' | 'typescript' | 'javascript' | 'java' | 'go' | 'cpp';
  category?: 'naming' | 'structure' | 'testing' | 'documentation' | 'security';
}

// 查询知识图谱参数
export interface QueryKnowledgeGraphParams {
  entity: string;
  relation_type?: 'CALLS' | 'INHERITS' | 'USES' | 'DEPENDS_ON' | 'IMPLEMENTS';
  depth?: number;
}

// RAG搜索结果
export interface RagSearchResult {
  id: string;
  title: string;
  content: string;
  file_path?: string;
  description?: string;
  code?: string;
  usage_context?: string;
  best_practices?: string[];
  related_docs?: string[];
  entities?: string[];
  score: number;
  metadata: Record<string, any>;
}

// 知识图谱实体
export interface GraphEntity {
  id: string;
  name: string;
  type: string;
  description?: string;
  properties: Record<string, any>;
}

// 知识图谱关系
export interface GraphRelation {
  id: string;
  source: string;
  target: string;
  type: string;
  description?: string;
  properties: Record<string, any>;
}

// 知识图谱查询结果
export interface GraphQueryResult {
  entity_info: GraphEntity;
  relationships: GraphRelation[];
  related_entities: GraphEntity[];
  usage_examples: Array<{
    context: string;
    code?: string;
    language?: string;
  }>;
}

// 编码规范结果
export interface CodingStandardsResult {
  language: string;
  naming_conventions: Record<string, string>;
  code_structure: Record<string, string>;
  best_practices: string[];
  code_templates: Record<string, string>;
  linting_rules: Record<string, any>;
  testing_guidelines: string[];
  tools: {
    formatters: string[];
    linters: string[];
    test_frameworks: string[];
  };
}

// 知识库统计
export interface KnowledgeBaseStats {
  total_documents: number;
  total_chunks: number;
  total_entities: number;
  total_relations: number;
  languages: string[];
  teams: string[];
  projects: string[];
  last_updated: string;
}

// API响应类型
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}