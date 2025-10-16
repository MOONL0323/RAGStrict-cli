/**
 * MCP服务器配置
 */

import dotenv from 'dotenv';
import path from 'path';

// 加载环境变量
dotenv.config();

export interface ServerConfig {
  // 服务器基本信息
  serverName: string;
  serverVersion: string;
  nodeEnv: string;

  // RAG引擎配置
  ragEngineUrl: string;
  ragEngineTimeout: number;

  // 后端API配置
  backendApiUrl: string;
  backendApiTimeout: number;

  // 缓存配置
  cacheTtl: number;
  cacheMaxKeys: number;

  // 日志配置
  logLevel: string;
  logFile: string;

  // 性能配置
  maxConcurrentRequests: number;
  rateLimitPerMinute: number;
}

export const config: ServerConfig = {
  // 服务器基本信息
  serverName: process.env.SERVER_NAME || 'team-context-server',
  serverVersion: process.env.SERVER_VERSION || '1.0.0',
  nodeEnv: process.env.NODE_ENV || 'development',

  // RAG引擎配置
  ragEngineUrl: process.env.RAG_ENGINE_URL || 'http://127.0.0.1:8000',
  ragEngineTimeout: parseInt(process.env.RAG_ENGINE_TIMEOUT || '30000'),

  // 后端API配置
  backendApiUrl: process.env.BACKEND_API_URL || 'http://127.0.0.1:8000/api/v1',
  backendApiTimeout: parseInt(process.env.BACKEND_API_TIMEOUT || '10000'),

  // 缓存配置
  cacheTtl: parseInt(process.env.CACHE_TTL || '300'), // 5分钟
  cacheMaxKeys: parseInt(process.env.CACHE_MAX_KEYS || '1000'),

  // 日志配置
  logLevel: process.env.LOG_LEVEL || 'info',
  logFile: process.env.LOG_FILE || path.join(process.cwd(), 'logs', 'mcp-server.log'),

  // 性能配置
  maxConcurrentRequests: parseInt(process.env.MAX_CONCURRENT_REQUESTS || '10'),
  rateLimitPerMinute: parseInt(process.env.RATE_LIMIT_PER_MINUTE || '100'),
};

// 验证配置
export function validateConfig(): void {
  const required = [
    'serverName',
    'ragEngineUrl',
    'backendApiUrl'
  ];

  for (const key of required) {
    if (!config[key as keyof ServerConfig]) {
      throw new Error(`缺少必需的配置项: ${key}`);
    }
  }

  // 验证URL格式
  try {
    new URL(config.ragEngineUrl);
    new URL(config.backendApiUrl);
  } catch (error) {
    throw new Error('无效的URL配置');
  }

  // 验证数值配置
  if (config.ragEngineTimeout < 1000) {
    throw new Error('RAG引擎超时时间不能小于1秒');
  }

  if (config.cacheTtl < 60) {
    throw new Error('缓存TTL不能小于60秒');
  }
}