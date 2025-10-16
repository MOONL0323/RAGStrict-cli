/**
 * RAGStrict MCP服务器
 * 为AI助手提供本地知识库访问能力
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';

import { DatabaseService } from './services/database-service.js';
import { logger } from './utils/logger.js';

class RAGStrictMCPServer {
  private server: Server;
  private dbService: DatabaseService;

  constructor() {
    this.server = new Server(
      {
        name: 'ragstrict-mcp',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    // 初始化数据库服务
    try {
      this.dbService = new DatabaseService();
    } catch (error) {
      logger.error('Failed to initialize database service', error);
      throw error;
    }

    this.setupToolHandlers();
    this.setupErrorHandlers();
  }

  /**
   * 设置工具处理器
   */
  private setupToolHandlers(): void {
    // 列出可用工具
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: 'search_documents',
            description: '在用户的知识库中搜索相关文档',
            inputSchema: {
              type: 'object',
              properties: {
                query: {
                  type: 'string',
                  description: '搜索查询'
                },
                limit: {
                  type: 'number',
                  default: 5,
                  description: '返回结果数量'
                }
              },
              required: ['query']
            }
          },
          {
            name: 'get_document',
            description: '获取指定文档的完整内容',
            inputSchema: {
              type: 'object',
              properties: {
                document_id: {
                  type: 'number',
                  description: '文档ID'
                }
              },
              required: ['document_id']
            }
          },
          {
            name: 'list_documents',
            description: '列出用户上传的所有文档',
            inputSchema: {
              type: 'object',
              properties: {
                limit: {
                  type: 'number',
                  default: 20,
                  description: '返回数量限制'
                },
                offset: {
                  type: 'number',
                  default: 0,
                  description: '偏移量'
                }
              }
            }
          },
          {
            name: 'get_stats',
            description: '获取知识库统计信息',
            inputSchema: {
              type: 'object',
              properties: {}
            }
          }
        ]
      };
    });

    // 处理工具调用
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case 'search_documents':
            return await this.handleSearchDocuments(args);
          
          case 'get_document':
            return await this.handleGetDocument(args);
          
          case 'list_documents':
            return await this.handleListDocuments(args);
          
          case 'get_stats':
            return await this.handleGetStats();
          
          default:
            throw new McpError(
              ErrorCode.MethodNotFound,
              `Unknown tool: ${name}`
            );
        }
      } catch (error) {
        logger.error(`Tool execution error: ${name}`, error);
        throw new McpError(
          ErrorCode.InternalError,
          `Tool execution failed: ${error instanceof Error ? error.message : 'Unknown error'}`
        );
      }
    });
  }

  /**
   * 处理搜索文档
   */
  private async handleSearchDocuments(args: any) {
    const { query, limit = 5 } = args;
    
    logger.info('Searching documents', { query, limit });
    
    const results = this.dbService.searchChunks(query, limit);
    
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            success: true,
            query,
            results: results.map(r => ({
              document_id: r.document_id,
              document_name: r.document_name,
              chunk_index: r.chunk_index,
              content: r.content,
              similarity: r.similarity
            })),
            total: results.length
          }, null, 2)
        }
      ]
    };
  }

  /**
   * 处理获取文档
   */
  private async handleGetDocument(args: any) {
    const { document_id } = args;
    
    logger.info('Getting document', { document_id });
    
    const document = this.dbService.getDocument(document_id);
    
    if (!document) {
      throw new McpError(
        ErrorCode.InvalidParams,
        `Document ${document_id} not found`
      );
    }
    
    const chunks = this.dbService.getDocumentChunks(document_id);
    
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            success: true,
            document: {
              id: document.id,
              filename: document.filename,
              filepath: document.filepath,
              file_type: document.file_type,
              file_size: document.file_size,
              status: document.status,
              created_at: document.created_at
            },
            chunks: chunks.map(c => ({
              index: c.chunk_index,
              content: c.content
            })),
            total_chunks: chunks.length
          }, null, 2)
        }
      ]
    };
  }

  /**
   * 处理列出文档
   */
  private async handleListDocuments(args: any) {
    const { limit = 20, offset = 0 } = args;
    
    logger.info('Listing documents', { limit, offset });
    
    const documents = this.dbService.listDocuments(limit, offset);
    
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            success: true,
            documents: documents.map(d => ({
              id: d.id,
              filename: d.filename,
              file_type: d.file_type,
              file_size: d.file_size,
              status: d.status,
              project_id: d.project_id,
              created_at: d.created_at
            })),
            total: documents.length,
            limit,
            offset
          }, null, 2)
        }
      ]
    };
  }

  /**
   * 处理获取统计
   */
  private async handleGetStats() {
    logger.info('Getting stats');
    
    const stats = this.dbService.getStats();
    
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            success: true,
            stats: {
              documents: stats.documents,
              chunks: stats.chunks,
              embeddings: stats.embeddings
            }
          }, null, 2)
        }
      ]
    };
  }

  /**
   * 设置错误处理器
   */
  private setupErrorHandlers(): void {
    this.server.onerror = (error) => {
      logger.error('Server error:', error);
    };

    process.on('SIGINT', async () => {
      logger.info('Shutting down server...');
      this.dbService.close();
      await this.server.close();
      process.exit(0);
    });
  }

  /**
   * 启动服务器
   */
  async start(): Promise<void> {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    logger.info('RAGStrict MCP Server started');
  }
}

// 启动服务器
const server = new RAGStrictMCPServer();
server.start().catch((error) => {
  logger.error('Failed to start server:', error);
  process.exit(1);
});
