"""
RAGStrict CLI - Command Line Interface for RAG Document Management
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from pathlib import Path
from typing import Optional
import asyncio
import sys

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragstrict.core.config import get_config, reset_config
from ragstrict.core.database import (
    init_database,
    create_tables,
    drop_tables,
    clean_database,
    get_db_session,
    run_async,
)
from ragstrict.services.document_processor import DocumentProcessor
from ragstrict.services.vector_search import VectorSearch
from ragstrict.services.stats_service import StatsService

app = typer.Typer(
    name="rags",
    help="RAGStrict - AI Context Enhancement Tool",
    add_completion=False,
)

console = Console()


@app.command()
def init(
    config_dir: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration directory (default: ./.ragstrict)"
    ),
):
    """Initialize RAGStrict project"""
    
    console.print(Panel.fit(
        "[bold cyan]Initializing RAGStrict...[/bold cyan]",
        border_style="cyan"
    ))
    
    try:
        # Initialize config
        if config_dir:
            reset_config()
        config = get_config(config_dir)
        
        # Initialize database
        init_database(config_dir)
        
        # Create tables
        run_async(create_tables())
        
        console.print(f"[green]✓[/green] Config directory: {config.data_dir.parent}")
        console.print(f"[green]✓[/green] Database: {config.db_path}")
        console.print(f"[green]✓[/green] Models: {config.models_dir}")
        console.print(f"[green]✓[/green] Uploads: {config.uploads_dir}")
        console.print("\n[bold green]Initialization complete![/bold green]")
        
    except Exception as e:
        console.print(f"[red]Error during initialization: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def add(
    path: Path = typer.Argument(..., help="File or directory to add"),
    project_id: Optional[int] = typer.Option(None, "--project", "-p", help="Project ID"),
    no_process: bool = typer.Option(False, "--no-process", help="Skip automatic processing"),
    config_dir: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Add documents to knowledge base"""
    
    if not path.exists():
        console.print(f"[red]Error: Path does not exist: {path}[/red]")
        raise typer.Exit(1)
    
    try:
        config = get_config(config_dir)
        init_database(config_dir)
        
        async def process_files():
            files = []
            if path.is_file():
                files = [path]
            else:
                # Recursively find all text files
                file_generator = path.glob("**/*")
                files = [f for f in file_generator if f.is_file()]
            
            if not files:
                console.print("[yellow]No files found to add[/yellow]")
                return
            
            console.print(f"[cyan]Found {len(files)} files[/cyan]\n")
            
            async with get_db_session() as session:
                processor = DocumentProcessor()
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Processing files...", total=len(files))
                    
                    for file in files:
                        try:
                            doc = await processor.add_document(
                                session,
                                file,
                                project_id=project_id,
                                auto_process=not no_process,
                            )
                            progress.console.print(
                                f"[green]✓[/green] {file.name} (ID: {doc.id})"
                            )
                        except Exception as e:
                            progress.console.print(
                                f"[red]✗[/red] {file.name}: {e}"
                            )
                        progress.advance(task)
            
            console.print("\n[bold green]Done![/bold green]")
        
        run_async(process_files())
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top", "-k", help="Number of results"),
    min_score: float = typer.Option(0.0, "--min-score", "-s", help="Minimum similarity score"),
    config_dir: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Search for similar documents"""
    
    try:
        config = get_config(config_dir)
        init_database(config_dir)
        
        async def do_search():
            async with get_db_session() as session:
                searcher = VectorSearch()
                results = await searcher.search(
                    session,
                    query,
                    top_k=top_k,
                    min_score=min_score,
                )
                
                if not results:
                    console.print("[yellow]No results found[/yellow]")
                    return
                
                console.print(f"\n[bold]Found {len(results)} results:[/bold]\n")
                
                for i, result in enumerate(results, 1):
                    console.print(Panel(
                        f"[cyan]{result['document_name']}[/cyan]\n"
                        f"Similarity: [green]{result['similarity']:.4f}[/green]\n"
                        f"Chunk {result['chunk_index']}\n\n"
                        f"{result['content'][:200]}...",
                        title=f"Result {i}",
                        border_style="cyan",
                    ))
        
        run_async(do_search())
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list(
    limit: int = typer.Option(50, "--limit", "-l", help="Number of documents to show"),
    offset: int = typer.Option(0, "--offset", "-o", help="Offset"),
    config_dir: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """List all documents"""
    
    try:
        config = get_config(config_dir)
        init_database(config_dir)
        
        async def get_documents():
            async with get_db_session() as session:
                stats_service = StatsService()
                docs = await stats_service.get_document_list(session, limit, offset)
                
                if not docs:
                    console.print("[yellow]No documents found[/yellow]")
                    return
                
                table = Table(title="Documents", show_header=True, header_style="bold cyan")
                table.add_column("ID", style="dim")
                table.add_column("Filename")
                table.add_column("Type")
                table.add_column("Size")
                table.add_column("Status")
                table.add_column("Created")
                
                for doc in docs:
                    size_kb = doc["file_size"] / 1024
                    created = doc["created_at"][:19] if doc["created_at"] else "N/A"
                    
                    table.add_row(
                        str(doc["id"]),
                        doc["filename"],
                        doc["file_type"],
                        f"{size_kb:.1f} KB",
                        doc["status"],
                        created,
                    )
                
                console.print("\n", table, "\n")
        
        run_async(get_documents())
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def stats(
    config_dir: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Show statistics"""
    
    try:
        config = get_config(config_dir)
        init_database(config_dir)
        
        async def get_stats():
            async with get_db_session() as session:
                stats_service = StatsService()
                stats = await stats_service.get_stats(session)
                
                # Create table
                table = Table(title="RAGStrict Statistics", show_header=True, header_style="bold cyan")
                table.add_column("Category", style="cyan")
                table.add_column("Metric", style="magenta")
                table.add_column("Value", justify="right", style="green")
                
                table.add_row("Documents", "Total", str(stats["documents"]["total"]))
                table.add_row("Chunks", "Total", str(stats["chunks"]["total"]))
                table.add_row(
                    "Embeddings",
                    "Total",
                    str(stats["embeddings"]["total"])
                )
                table.add_row(
                    "Embeddings",
                    "Vectorization Rate",
                    f"{stats['embeddings']['vectorization_rate']:.1f}%"
                )
                table.add_row(
                    "Knowledge Graph",
                    "Entities",
                    str(stats["knowledge_graph"]["entities"])
                )
                table.add_row(
                    "Knowledge Graph",
                    "Relations",
                    str(stats["knowledge_graph"]["relations"])
                )
                
                console.print("\n", table, "\n")
        
        run_async(get_stats())
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def mcp(
    port: int = typer.Option(3000, "--port", "-p", help="MCP server port"),
    config_dir: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Start MCP server"""
    
    console.print(Panel.fit(
        f"[bold cyan]Starting MCP Server on port {port}...[/bold cyan]",
        border_style="cyan"
    ))
    
    try:
        import subprocess
        from pathlib import Path
        
        # MCP server directory
        mcp_dir = Path(__file__).parent.parent.parent / "mcp-server"
        
        if not mcp_dir.exists():
            console.print(f"[red]Error: MCP server directory not found: {mcp_dir}[/red]")
            raise typer.Exit(1)
        
        # Start MCP server
        console.print(f"[cyan]MCP directory: {mcp_dir}[/cyan]")
        console.print("[cyan]Running: npm run build && npm start[/cyan]\n")
        
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=mcp_dir,
            shell=True,
        )
        
        if result.returncode != 0:
            console.print("[red]Failed to build MCP server[/red]")
            raise typer.Exit(1)
        
        # Start server
        subprocess.run(
            ["npm", "start"],
            cwd=mcp_dir,
            shell=True,
        )
        
    except KeyboardInterrupt:
        console.print("\n[yellow]MCP server stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def clean(
    config_dir: Optional[Path] = typer.Option(None, "--config", "-c"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Clean all data from database"""
    
    if not confirm:
        confirm = typer.confirm(
            "This will delete all documents, chunks, and embeddings. Continue?",
            default=False,
        )
    
    if not confirm:
        console.print("[yellow]Cancelled[/yellow]")
        return
    
    try:
        config = get_config(config_dir)
        init_database(config_dir)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Cleaning database...", total=None)
            run_async(clean_database())
            progress.update(task, completed=True)
        
        console.print("[green]Database cleaned successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def show(
    doc_id: int = typer.Argument(..., help="Document ID"),
    config_dir: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Show document details"""
    
    try:
        config = get_config(config_dir)
        init_database(config_dir)
        
        async def get_document():
            async with get_db_session() as session:
                from sqlalchemy import select
                from ragstrict.core.database import Document, Chunk
                
                # Get document
                result = await session.execute(
                    select(Document).where(Document.id == doc_id)
                )
                doc = result.scalar_one_or_none()
                
                if not doc:
                    console.print(f"[red]Document with ID {doc_id} not found[/red]")
                    return
                
                # Get chunks count
                result = await session.execute(
                    select(Chunk).where(Chunk.document_id == doc_id)
                )
                chunks = result.scalars().all()
                
                # Display document info
                console.print(f"\n[bold cyan]Document #{doc.id}[/bold cyan]\n")
                console.print(f"[cyan]Filename:[/cyan] {doc.filename}")
                console.print(f"[cyan]Path:[/cyan] {doc.filepath}")
                console.print(f"[cyan]Type:[/cyan] {doc.file_type}")
                console.print(f"[cyan]Size:[/cyan] {doc.file_size / 1024:.2f} KB")
                console.print(f"[cyan]Status:[/cyan] {doc.status}")
                console.print(f"[cyan]Chunks:[/cyan] {len(chunks)}")
                console.print(f"[cyan]Created:[/cyan] {doc.created_at}")
                
                if doc.project_id:
                    console.print(f"[cyan]Project ID:[/cyan] {doc.project_id}")
                
                # Show first chunk preview
                if chunks:
                    console.print(f"\n[bold]First Chunk Preview:[/bold]")
                    preview = chunks[0].content[:200] + "..." if len(chunks[0].content) > 200 else chunks[0].content
                    console.print(Panel(preview, border_style="cyan"))
        
        run_async(get_document())
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def delete(
    doc_id: int = typer.Argument(..., help="Document ID to delete"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    config_dir: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Delete document and its embeddings"""
    
    try:
        config = get_config(config_dir)
        init_database(config_dir)
        
        async def delete_document():
            async with get_db_session() as session:
                from sqlalchemy import select, delete as sql_delete
                from ragstrict.core.database import Document, Chunk, VectorEmbedding
                
                # Get document
                result = await session.execute(
                    select(Document).where(Document.id == doc_id)
                )
                doc = result.scalar_one_or_none()
                
                if not doc:
                    console.print(f"[red]Document with ID {doc_id} not found[/red]")
                    return
                
                if not confirm:
                    confirm_delete = typer.confirm(
                        f"Delete document '{doc.filename}' and all its data?",
                        default=False
                    )
                    if not confirm_delete:
                        console.print("[yellow]Cancelled[/yellow]")
                        return
                
                # Delete embeddings
                await session.execute(
                    sql_delete(VectorEmbedding).where(
                        VectorEmbedding.chunk_id.in_(
                            select(Chunk.id).where(Chunk.document_id == doc_id)
                        )
                    )
                )
                
                # Delete chunks
                await session.execute(
                    sql_delete(Chunk).where(Chunk.document_id == doc_id)
                )
                
                # Delete document
                await session.execute(
                    sql_delete(Document).where(Document.id == doc_id)
                )
                
                await session.commit()
                
                console.print(f"[green]✓[/green] Deleted document '{doc.filename}' and all associated data")
        
        run_async(delete_document())
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def tree(
    config_dir: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Show documents in tree structure"""
    
    try:
        config = get_config(config_dir)
        init_database(config_dir)
        
        async def show_tree():
            async with get_db_session() as session:
                from sqlalchemy import select
                from ragstrict.core.database import Document
                from collections import defaultdict
                
                # Get all documents
                result = await session.execute(select(Document))
                docs = result.scalars().all()
                
                if not docs:
                    console.print("[yellow]No documents found[/yellow]")
                    return
                
                # Group by project_id
                by_project = defaultdict(list)
                for doc in docs:
                    by_project[doc.project_id or "None"].append(doc)
                
                console.print("\n[bold cyan]Document Tree[/bold cyan]\n")
                
                for project_id, project_docs in sorted(by_project.items()):
                    if project_id == "None":
                        console.print("[cyan]├── No Project[/cyan]")
                    else:
                        console.print(f"[cyan]├── Project {project_id}[/cyan]")
                    
                    for i, doc in enumerate(project_docs):
                        is_last = i == len(project_docs) - 1
                        prefix = "    └──" if is_last else "    ├──"
                        size_kb = doc.file_size / 1024
                        console.print(f"{prefix} {doc.filename} ({size_kb:.1f} KB) [dim]#{doc.id}[/dim]")
                
                console.print()
        
        run_async(show_tree())
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def chat(
    query: str = typer.Argument(..., help="Your question or query"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of documents to retrieve"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Enter interactive mode")
):
    """Chat with your documents using RAG-enhanced context
    
    Examples:
      rags chat "What is the main topic of the documents?"
      rags chat "Summarize the key points" --top-k 5
      rags chat --interactive  # Enter interactive mode
    """
    try:
        from ragstrict.core.database import run_async
        from ragstrict.core.config import get_config
        from ragstrict.services.vector_search import VectorSearch
        from ragstrict.services.embedding_service import EmbeddingService
        import aiohttp
        
        config = get_config()
        
        # Check if API is configured
        if not config.enable_api or not config.llm_api_url:
            console.print("[yellow]⚠️  LLM API not configured. Please create .env.api with ENABLE_API=true and LLM_API_URL.[/yellow]")
            raise typer.Exit(1)
        
        async def chat_once(question: str):
            """Process single query"""
            # Search for relevant documents
            search_service = VectorSearch(EmbeddingService())
            results = await search_service.search(question, top_k=top_k)
            
            if not results:
                console.print("[yellow]No relevant documents found. Please add documents first with 'rags add'.[/yellow]")
                return None
            
            # Build context from retrieved documents
            console.print(f"[dim]📚 Retrieved {len(results)} relevant documents[/dim]")
            context_parts = []
            for i, (doc, chunk, score) in enumerate(results, 1):
                context_parts.append(f"Document {i}: {doc.filename}\n{chunk.content}\n")
            
            context = "\n---\n".join(context_parts)
            
            # Call LLM API
            headers = {
                "Content-Type": "application/json",
                "Authorization": config.llm_api_key or "",
            }
            
            system_prompt = """You are a helpful AI assistant. Answer questions based on the provided context.
If the context doesn't contain enough information, say so clearly."""
            
            user_prompt = f"""Context:
{context}

Question: {question}

Please provide a clear and concise answer based on the context above."""
            
            payload = {
                "model": config.llm_api_model or "qwen3-32b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000,
            }
            
            console.print("\n[cyan]💭 Thinking...[/cyan]")
            
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config.llm_api_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        console.print(f"[red]❌ API Error {response.status}: {error_text}[/red]")
                        return None
                    
                    result = await response.json()
                    
                    # Parse response
                    if "choices" in result and len(result["choices"]) > 0:
                        answer = result["choices"][0]["message"]["content"]
                        return answer
                    else:
                        console.print(f"[red]❌ Unexpected API response format[/red]")
                        return None
        
        async def interactive_chat():
            """Interactive chat mode"""
            console.print("[bold cyan]🤖 RAGStrict Interactive Chat[/bold cyan]")
            console.print("[dim]Type 'exit' or 'quit' to leave, 'clear' to clear screen[/dim]\n")
            
            while True:
                try:
                    question = console.input("[bold green]You:[/bold green] ")
                    
                    if not question.strip():
                        continue
                    
                    if question.lower() in ["exit", "quit"]:
                        console.print("[dim]Goodbye! 👋[/dim]")
                        break
                    
                    if question.lower() == "clear":
                        console.clear()
                        continue
                    
                    answer = await chat_once(question)
                    
                    if answer:
                        console.print(f"\n[bold cyan]Assistant:[/bold cyan] {answer}\n")
                
                except KeyboardInterrupt:
                    console.print("\n[dim]Goodbye! 👋[/dim]")
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]\n")
        
        if interactive:
            run_async(interactive_chat())
        else:
            answer = run_async(chat_once(query))
            if answer:
                console.print(f"\n[bold cyan]Answer:[/bold cyan]\n{answer}\n")
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def config(
    action: str = typer.Argument(..., help="Action: set, get, show"),
    key: str = typer.Argument(None, help="Config key (for set/get)"),
    value: str = typer.Argument(None, help="Config value (for set)")
):
    """Manage API configuration
    
    Actions:
      show              - Show all configuration
      get <key>         - Get specific config value
      set <key> <value> - Set config value
    
    Available keys:
      enable_api        - Enable/disable API (true/false)
      llm_api_url       - LLM API endpoint URL
      llm_api_key       - LLM API key
      llm_api_model     - LLM model name
      embedding_api_url - Embedding API endpoint URL
      embedding_api_key - Embedding API key
      embedding_api_model - Embedding model name
    
    Examples:
      rags config show
      rags config get llm_api_url
      rags config set enable_api true
      rags config set llm_api_url https://api.example.com/v1/chat/completions
    """
    try:
        from ragstrict.core.config import get_config
        from pathlib import Path
        import os
        
        config = get_config()
        api_env_file = config.data_dir.parent / ".env.api"
        
        if action == "show":
            console.print("[bold cyan]API Configuration:[/bold cyan]\n")
            
            if not api_env_file.exists():
                console.print("[yellow]No .env.api file found. API features are disabled.[/yellow]")
                console.print(f"\nCreate file: {api_env_file}")
                console.print("Or use: rags config set <key> <value>")
                return
            
            # Read and display config
            with open(api_env_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            console.print(f"[dim]Config file: {api_env_file}[/dim]\n")
            
            # Parse and display key values
            from rich.table import Table
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")
            
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    # Mask sensitive values
                    if 'key' in k.lower() and v and len(v) > 10:
                        v = v[:6] + "..." + v[-4:]
                    table.add_row(k, v)
            
            console.print(table)
        
        elif action == "get":
            if not key:
                console.print("[red]Error: key required for 'get' action[/red]")
                console.print("Usage: rags config get <key>")
                raise typer.Exit(1)
            
            if not api_env_file.exists():
                console.print(f"[yellow].env.api not found at: {api_env_file}[/yellow]")
                raise typer.Exit(1)
            
            # Read config file
            with open(api_env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        if k.strip().upper() == key.upper():
                            console.print(f"[cyan]{key}[/cyan] = [green]{v}[/green]")
                            return
            
            console.print(f"[yellow]Key '{key}' not found in .env.api[/yellow]")
        
        elif action == "set":
            if not key or value is None:
                console.print("[red]Error: key and value required for 'set' action[/red]")
                console.print("Usage: rags config set <key> <value>")
                raise typer.Exit(1)
            
            key_upper = key.upper()
            
            # Validate key
            valid_keys = [
                "ENABLE_API", "LLM_API_URL", "LLM_API_KEY", "LLM_API_MODEL",
                "EMBEDDING_API_URL", "EMBEDDING_API_KEY", "EMBEDDING_API_MODEL",
                "API_TIMEOUT", "LLM_TIMEOUT"
            ]
            
            if key_upper not in valid_keys:
                console.print(f"[yellow]Warning: '{key}' is not a standard config key[/yellow]")
            
            # Create .env.api if not exists
            if not api_env_file.exists():
                console.print(f"[yellow]Creating .env.api at: {api_env_file}[/yellow]")
                api_env_file.parent.mkdir(parents=True, exist_ok=True)
                api_env_file.write_text("# RAGStrict API Configuration\n\n", encoding="utf-8")
            
            # Read existing content
            with open(api_env_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Update or add the key
            key_found = False
            new_lines = []
            
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith('#') and '=' in stripped:
                    existing_key = stripped.split('=', 1)[0].strip()
                    if existing_key.upper() == key_upper:
                        new_lines.append(f"{key_upper}={value}\n")
                        key_found = True
                        continue
                new_lines.append(line)
            
            # Add new key if not found
            if not key_found:
                new_lines.append(f"{key_upper}={value}\n")
            
            # Write back
            with open(api_env_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            
            console.print(f"[green]Set {key_upper} = {value}[/green]")
            console.print(f"[dim]Config saved to: {api_env_file}[/dim]")
        
        else:
            console.print(f"[red]Unknown action: {action}[/red]")
            console.print("Valid actions: show, get, set")
            raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def help():
    """Show detailed help for all commands"""
    console.print("[bold cyan]RAGStrict - 命令行帮助[/bold cyan]\n")
    
    commands = {
        "init": {
            "desc": "初始化数据库和配置目录",
            "usage": "rags init",
            "examples": ["rags init"]
        },
        "add": {
            "desc": "添加文档到知识库",
            "usage": "rags add <file-or-directory>",
            "examples": [
                "rags add document.txt",
                "rags add /path/to/directory"
            ]
        },
        "search": {
            "desc": "语义搜索文档",
            "usage": "rags search <query> [--top-k N]",
            "examples": [
                "rags search '如何使用'",
                "rags search 'python async' --top-k 5"
            ]
        },
        "list": {
            "desc": "列出所有文档",
            "usage": "rags list [--verbose]",
            "examples": [
                "rags list",
                "rags list --verbose"
            ]
        },
        "show": {
            "desc": "显示文档详情",
            "usage": "rags show <document-id>",
            "examples": ["rags show 1"]
        },
        "delete": {
            "desc": "删除文档",
            "usage": "rags delete <document-id>",
            "examples": ["rags delete 1"]
        },
        "clean": {
            "desc": "清空整个数据库",
            "usage": "rags clean",
            "examples": ["rags clean"]
        },
        "stats": {
            "desc": "显示统计信息",
            "usage": "rags stats",
            "examples": ["rags stats"]
        },
        "chat": {
            "desc": "与文档对话 (需要配置API)",
            "usage": "rags chat <query> [--top-k N] [--interactive]",
            "examples": [
                "rags chat '这个项目是做什么的?'",
                "rags chat --interactive"
            ]
        },
        "config": {
            "desc": "管理API配置",
            "usage": "rags config <show|get|set> [key] [value]",
            "examples": [
                "rags config show",
                "rags config get llm_api_url",
                "rags config set enable_api true"
            ]
        },
        "tree": {
            "desc": "显示文档树结构",
            "usage": "rags tree",
            "examples": ["rags tree"]
        },
        "mcp": {
            "desc": "启动MCP服务器",
            "usage": "rags mcp",
            "examples": ["rags mcp"]
        },
        "version": {
            "desc": "显示版本信息",
            "usage": "rags version",
            "examples": ["rags version"]
        },
        "help": {
            "desc": "显示此帮助信息",
            "usage": "rags help",
            "examples": ["rags help"]
        }
    }
    
    for cmd, info in commands.items():
        console.print(f"[bold green]{cmd}[/bold green]")
        console.print(f"  {info['desc']}")
        console.print(f"  [dim]用法: {info['usage']}[/dim]")
        console.print(f"  [dim]示例:[/dim]")
        for example in info['examples']:
            console.print(f"    [cyan]{example}[/cyan]")
        console.print()
    
    console.print("[bold yellow]快速开始:[/bold yellow]")
    console.print("  1. rags init                    # 初始化")
    console.print("  2. rags add document.txt        # 添加文档")
    console.print("  3. rags search '你的问题'       # 搜索")
    console.print("  4. rags config show             # 查看配置")
    console.print()
    console.print("[dim]详细文档: README.md[/dim]")


@app.command()
def version():
    """Show version information"""
    from ragstrict import __version__
    
    console.print(f"[bold cyan]RAGStrict[/bold cyan] version [green]{__version__}[/green]")
    console.print("AI Context Enhancement Tool")
    console.print("https://github.com/MOONL0323/RAGStrict")


if __name__ == "__main__":
    app()
