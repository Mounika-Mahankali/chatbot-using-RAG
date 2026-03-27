#!/usr/bin/env python3
"""
Migration script for upgrading to Persistent RAG system
This script helps migrate existing chat sessions to use persistent embeddings
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import json

def migrate_to_persistent_rag():
    """
    Migrate existing database to support persistent RAG
    """
    
    db_path = Path(__file__).parent / "chatbot.db"
    
    if not db_path.exists():
        print("⚠️ No existing database found. Starting fresh with new schema.")
        return True
    
    print("🔄 Migrating to Persistent RAG system...")
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(chat_sessions)")
        columns = [col[1] for col in cursor.fetchall()]
        
        migrations_needed = []
        
        if 'has_embeddings' not in columns:
            migrations_needed.append("adding has_embeddings column")
            cursor.execute("""
                ALTER TABLE chat_sessions 
                ADD COLUMN has_embeddings INTEGER DEFAULT 0
            """)
            print("  ✓ Added 'has_embeddings' column")
        
        if 'embeddings_updated_at' not in columns:
            migrations_needed.append("adding embeddings_updated_at column")
            cursor.execute("""
                ALTER TABLE chat_sessions 
                ADD COLUMN embeddings_updated_at DATETIME
            """)
            print("  ✓ Added 'embeddings_updated_at' column")
        
        if not migrations_needed:
            print("  ✓ Database already has all required columns")
        
        conn.commit()
        
        # Show migration summary
        cursor.execute("SELECT COUNT(*) FROM chat_sessions")
        chat_count = cursor.fetchone()[0]
        
        print(f"\n✅ Migration complete!")
        print(f"   • Total chat sessions: {chat_count}")
        print(f"   • All sessions will use persistent RAG on next embedding creation")
        print(f"   • Vector indices will be stored in: vector_indices/session_<id>/")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Migration error: {e}")
        return False


def create_vector_indices_dir():
    """Create vector_indices directory if it doesn't exist"""
    vector_dir = Path(__file__).parent / "vector_indices"
    vector_dir.mkdir(exist_ok=True)
    print(f"✓ Vector indices directory ready: {vector_dir}")


def verify_setup():
    """Verify the new system is set up correctly"""
    print("\n🔍 Verifying setup...")
    
    checks = []
    
    # Check vector_db.py exists
    if (Path(__file__).parent / "vector_db.py").exists():
        checks.append(("vector_db.py module", True))
    else:
        checks.append(("vector_db.py module", False))
    
    # Check vector_indices directory
    if (Path(__file__).parent / "vector_indices").exists():
        checks.append(("vector_indices directory", True))
    else:
        checks.append(("vector_indices directory", False))
    
    # Check database
    if (Path(__file__).parent / "chatbot.db").exists():
        checks.append(("chatbot.db database", True))
    else:
        checks.append(("chatbot.db database", False))
    
    # Check required packages
    try:
        import langchain_community
        checks.append(("langchain_community", True))
    except ImportError:
        checks.append(("langchain_community", False))
    
    try:
        import faiss
        checks.append(("FAISS", True))
    except ImportError:
        checks.append(("FAISS", False))
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        checks.append(("HuggingFaceEmbeddings", True))
    except ImportError:
        checks.append(("HuggingFaceEmbeddings", False))
    
    print()
    all_ok = True
    for name, status in checks:
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {name}")
        if not status:
            all_ok = False
    
    return all_ok


def main():
    """Run the migration"""
    print("=" * 60)
    print("🚀 Persistent RAG Migration Tool")
    print("=" * 60)
    print()
    
    # Create directory
    create_vector_indices_dir()
    
    # Migrate database
    print()
    if not migrate_to_persistent_rag():
        print("\n❌ Migration failed!")
        return 1
    
    # Verify setup
    print()
    if not verify_setup():
        print("\n⚠️ Some dependencies might be missing.")
        print("Install required packages with:")
        print("  pip install langchain-community faiss-cpu langchain-huggingface")
        print()
    
    print("\n" + "=" * 60)
    print("✅ Setup complete! Your chatbot is now using Persistent RAG:")
    print("=" * 60)
    print()
    print("📌 Key improvements:")
    print("  • Embeddings created once and stored permanently")
    print("  • Future queries use cached embeddings (much faster)")
    print("  • Each chat session has its own vector database")
    print("  • Automatic cleanup when chats are deleted")
    print()
    print("📂 Vector indices stored in: vector_indices/")
    print("💾 Database: chatbot.db")
    print()
    print("To start the app: streamlit run app.py")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
