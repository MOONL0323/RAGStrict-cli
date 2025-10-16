/**
 * 日志工具
 */

import winston from 'winston';
import { config } from '../config/config.js';
import path from 'path';
import fs from 'fs';

// 确保日志目录存在
const logDir = path.dirname(config.logFile);
if (!fs.existsSync(logDir)) {
  fs.mkdirSync(logDir, { recursive: true });
}

// 创建Winston logger
export const logger = winston.createLogger({
  level: config.logLevel,
  format: winston.format.combine(
    winston.format.timestamp({
      format: 'YYYY-MM-DD HH:mm:ss'
    }),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { 
    service: config.serverName,
    version: config.serverVersion 
  },
  transports: [
    // 控制台输出
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple(),
        winston.format.printf(({ timestamp, level, message, service, ...meta }) => {
          let logMessage = `${timestamp} [${service}] ${level}: ${message}`;
          
          // 添加额外信息
          if (Object.keys(meta).length > 0) {
            logMessage += ` ${JSON.stringify(meta)}`;
          }
          
          return logMessage;
        })
      )
    }),
    
    // 文件输出
    new winston.transports.File({
      filename: config.logFile,
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
      ),
      maxsize: 5242880, // 5MB
      maxFiles: 5,
    }),
    
    // 错误日志单独文件
    new winston.transports.File({
      filename: path.join(logDir, 'error.log'),
      level: 'error',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
      ),
      maxsize: 5242880, // 5MB
      maxFiles: 5,
    })
  ],
});

// 在生产环境中，不输出debug级别的日志到控制台
if (config.nodeEnv === 'production') {
  logger.remove(logger.transports[0]); // 移除控制台输出
  logger.add(new winston.transports.Console({
    level: 'info',
    format: winston.format.combine(
      winston.format.timestamp(),
      winston.format.simple()
    )
  }));
}

// 创建子logger的辅助函数
export function createChildLogger(module: string): winston.Logger {
  return logger.child({ module });
}

// 性能监控工具
export class PerformanceMonitor {
  private static timers: Map<string, number> = new Map();

  static start(operation: string): void {
    this.timers.set(operation, Date.now());
  }

  static end(operation: string): number {
    const start = this.timers.get(operation);
    if (!start) {
      logger.warn(`性能监控: 未找到操作 ${operation} 的开始时间`);
      return 0;
    }

    const duration = Date.now() - start;
    this.timers.delete(operation);
    
    logger.info(`性能监控: ${operation} 耗时 ${duration}ms`);
    return duration;
  }

  static measure<T>(operation: string, fn: () => Promise<T>): Promise<T> {
    return new Promise(async (resolve, reject) => {
      this.start(operation);
      try {
        const result = await fn();
        this.end(operation);
        resolve(result);
      } catch (error) {
        this.end(operation);
        reject(error);
      }
    });
  }
}

// 请求跟踪工具
export class RequestTracker {
  private static requestCount = 0;
  private static activeRequests = 0;

  static startRequest(requestId?: string): string {
    const id = requestId || `req_${++this.requestCount}`;
    this.activeRequests++;
    
    logger.debug('请求开始', { 
      requestId: id, 
      activeRequests: this.activeRequests 
    });
    
    return id;
  }

  static endRequest(requestId: string, success: boolean = true): void {
    this.activeRequests--;
    
    logger.debug('请求结束', { 
      requestId, 
      success,
      activeRequests: this.activeRequests 
    });
  }

  static getActiveRequestCount(): number {
    return this.activeRequests;
  }
}

export default logger;