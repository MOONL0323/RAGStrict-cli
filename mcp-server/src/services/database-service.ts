/**
 * SQLite数据库服务 - 直接读取本地知识库
 */

import Database from 'better-sqlite3';
import path from 'path';
import { logger } from '../utils/logger.js';

interface Document {
  id: number;
  filename: string;
  filepath: string;
  file_type: string;
  file_size: number;
  project_id: number | null;
  status: string;
  created_at: string;
}

interface Chunk {
  id: number;
  document_id: number;
  content: string;
  chunk_index: number;
  chunk_metadata: string;
}

interface VectorEmbedding {
  id: number;
  chunk_id: number;
  embedding_vector: Buffer;
  model_name: string;
}

interface SearchResult {
  document_id: number;
  document_name: string;
  chunk_id: number;
  chunk_index: number;
  content: string;
  similarity: number;
}

export class DatabaseService {
  private db: Database.Database;
  private dbPath: string;

  constructor(dbPath?: string) {
    // 默认数据库路径
    this.dbPath = dbPath || path.join(process.cwd(), '.ragstrict', 'data', 'ragstrict.db');
    
    try {
      this.db = new Database(this.dbPath, { 
        readonly: true,  // 只读模式
        fileMustExist: true 
      });
      logger.info(`Connected to database: ${this.dbPath}`);
    } catch (error) {
      logger.error(`Failed to connect to database: ${this.dbPath}`, error);
      throw new Error(`Database connection failed: ${error}`);
    }
  }

  /**
   * 列出所有文档
   */
  listDocuments(limit: number = 50, offset: number = 0): Document[] {
    try {
      const stmt = this.db.prepare(`
        SELECT id, filename, filepath, file_type, file_size, project_id, status, created_at
        FROM documents
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
      `);
      
      return stmt.all(limit, offset) as Document[];
    } catch (error) {
      logger.error('Failed to list documents', error);
      throw error;
    }
  }

  /**
   * 获取文档详情
   */
  getDocument(documentId: number): Document | null {
    try {
      const stmt = this.db.prepare(`
        SELECT id, filename, filepath, file_type, file_size, project_id, status, created_at
        FROM documents
        WHERE id = ?
      `);
      
      return stmt.get(documentId) as Document | null;
    } catch (error) {
      logger.error(`Failed to get document ${documentId}`, error);
      throw error;
    }
  }

  /**
   * 获取文档的所有chunks
   */
  getDocumentChunks(documentId: number): Chunk[] {
    try {
      const stmt = this.db.prepare(`
        SELECT id, document_id, content, chunk_index, chunk_metadata
        FROM chunks
        WHERE document_id = ?
        ORDER BY chunk_index ASC
      `);
      
      return stmt.all(documentId) as Chunk[];
    } catch (error) {
      logger.error(`Failed to get chunks for document ${documentId}`, error);
      throw error;
    }
  }

  /**
   * 语义搜索 - 简化版本，返回所有chunks并在应用层计算相似度
   * 注意：这需要在Node.js中实现向量相似度计算，或者调用Python服务
   */
  searchChunks(query: string, limit: number = 5): SearchResult[] {
    try {
      // 简单的关键词搜索作为降级方案
      const stmt = this.db.prepare(`
        SELECT 
          c.document_id,
          d.filename as document_name,
          c.id as chunk_id,
          c.chunk_index,
          c.content,
          1.0 as similarity
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.content LIKE ?
        ORDER BY c.document_id, c.chunk_index
        LIMIT ?
      `);
      
      return stmt.all(`%${query}%`, limit) as SearchResult[];
    } catch (error) {
      logger.error('Failed to search chunks', error);
      throw error;
    }
  }

  /**
   * 获取统计信息
   */
  getStats(): {
    documents: number;
    chunks: number;
    embeddings: number;
  } {
    try {
      const docCount = this.db.prepare('SELECT COUNT(*) as count FROM documents').get() as { count: number };
      const chunkCount = this.db.prepare('SELECT COUNT(*) as count FROM chunks').get() as { count: number };
      const embeddingCount = this.db.prepare('SELECT COUNT(*) as count FROM vector_embeddings').get() as { count: number };
      
      return {
        documents: docCount.count,
        chunks: chunkCount.count,
        embeddings: embeddingCount.count
      };
    } catch (error) {
      logger.error('Failed to get stats', error);
      throw error;
    }
  }

  /**
   * 根据项目ID获取文档
   */
  getDocumentsByProject(projectId: number): Document[] {
    try {
      const stmt = this.db.prepare(`
        SELECT id, filename, filepath, file_type, file_size, project_id, status, created_at
        FROM documents
        WHERE project_id = ?
        ORDER BY created_at DESC
      `);
      
      return stmt.all(projectId) as Document[];
    } catch (error) {
      logger.error(`Failed to get documents for project ${projectId}`, error);
      throw error;
    }
  }

  /**
   * 关闭数据库连接
   */
  close(): void {
    try {
      this.db.close();
      logger.info('Database connection closed');
    } catch (error) {
      logger.error('Failed to close database', error);
    }
  }
}
