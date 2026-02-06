import nltk
import json
import re
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any

nltk.download("punkt")
nltk.download("punkt_tab")

# Configuration
CHUNK_SIZE = 300  # Target words per chunk
OVERLAP = 50
MIN_CHUNK_SIZE = 100  # Minimum viable chunk size

class SemanticChunker:
    """Advanced chunker that preserves document structure and handles multimedia content"""
    
    def __init__(self, chunk_size=300, overlap=50, min_chunk_size=100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
    
    def parse_wikipedia_content(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse Wikipedia content into structured blocks (headings, paragraphs, tables, images)
        """
        blocks = []
        
        # Regex patterns for different content types
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        table_pattern = r'\|.*\|'  # Simple table detection
        image_pattern = r'!\[([^\]]*)\]\(([^\)]+)\)|<img[^>]+src="([^"]+)"[^>]*>'
        
        lines = text.split('\n')
        current_section = None
        current_block = {'type': 'paragraph', 'content': '', 'metadata': {}}
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for headings
            heading_match = re.match(heading_pattern, line)
            if heading_match:
                # Save previous block
                if current_block['content'].strip():
                    blocks.append(current_block)
                
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2)
                current_section = heading_text
                
                blocks.append({
                    'type': 'heading',
                    'level': level,
                    'content': heading_text,
                    'metadata': {'section': current_section}
                })
                current_block = {'type': 'paragraph', 'content': '', 'metadata': {'section': current_section}}
                i += 1
                continue
            
            # Check for images
            image_match = re.search(image_pattern, line)
            if image_match:
                # Save previous block
                if current_block['content'].strip():
                    blocks.append(current_block)
                
                alt_text = image_match.group(1) or ''
                img_url = image_match.group(2) or image_match.group(3) or ''
                
                blocks.append({
                    'type': 'image',
                    'content': alt_text,
                    'metadata': {
                        'image_url': img_url,
                        'section': current_section,
                        'description': alt_text
                    }
                })
                current_block = {'type': 'paragraph', 'content': '', 'metadata': {'section': current_section}}
                i += 1
                continue
            
            # Check for tables (multi-line detection)
            if '|' in line and line.count('|') >= 2:
                # Save previous block
                if current_block['content'].strip():
                    blocks.append(current_block)
                
                # Collect table rows
                table_lines = [line]
                i += 1
                while i < len(lines) and '|' in lines[i]:
                    table_lines.append(lines[i].strip())
                    i += 1
                
                blocks.append({
                    'type': 'table',
                    'content': self._parse_table(table_lines),
                    'metadata': {
                        'section': current_section,
                        'raw_table': '\n'.join(table_lines)
                    }
                })
                current_block = {'type': 'paragraph', 'content': '', 'metadata': {'section': current_section}}
                continue
            
            # Regular paragraph text
            if line:
                if current_block['content']:
                    current_block['content'] += ' ' + line
                else:
                    current_block['content'] = line
                    current_block['metadata']['section'] = current_section
            
            i += 1
        
        # Add final block
        if current_block['content'].strip():
            blocks.append(current_block)
        
        return blocks
    
    def _parse_table(self, table_lines: List[str]) -> str:
        """Convert table to a textual representation"""
        # Extract headers and rows
        rows = []
        for line in table_lines:
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if cells:
                rows.append(cells)
        
        if not rows:
            return ""
        
        # Create textual representation
        if len(rows) > 0:
            headers = rows[0]
            text_parts = [f"Table with columns: {', '.join(headers)}"]
            
            for row in rows[1:]:
                if row and len(row) == len(headers):
                    row_text = ". ".join([f"{headers[i]}: {row[i]}" for i in range(len(headers))])
                    text_parts.append(row_text)
            
            return ". ".join(text_parts)
        
        return ""
    
    def create_chunks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create semantic chunks from structured blocks while preserving context
        """
        chunks = []
        current_chunk = {
            'text': '',
            'metadata': {
                'sections': [],
                'has_table': False,
                'has_image': False,
                'images': [],
                'tables': []
            }
        }
        current_words = 0
        
        for block in blocks:
            block_text = block['content']
            block_words = len(block_text.split())
            
            # Handle headings - always start new chunk
            if block['type'] == 'heading':
                if current_chunk['text'].strip():
                    chunks.append(current_chunk.copy())
                
                current_chunk = {
                    'text': f"[SECTION: {block_text}]",
                    'metadata': {
                        'sections': [block_text],
                        'has_table': False,
                        'has_image': False,
                        'images': [],
                        'tables': []
                    }
                }
                current_words = len(block_text.split())
                continue
            
            # Handle images
            if block['type'] == 'image':
                image_text = f"[IMAGE: {block_text}]" if block_text else "[IMAGE]"
                current_chunk['text'] += f" {image_text}"
                current_chunk['metadata']['has_image'] = True
                current_chunk['metadata']['images'].append({
                    'url': block['metadata'].get('image_url', ''),
                    'description': block_text
                })
                current_words += len(image_text.split())
                continue
            
            # Handle tables
            if block['type'] == 'table':
                table_text = f"[TABLE] {block_text}"
                
                # Tables often need their own chunk
                if current_words + len(table_text.split()) > self.chunk_size and current_chunk['text'].strip():
                    chunks.append(current_chunk.copy())
                    current_chunk = {
                        'text': table_text,
                        'metadata': {
                            'sections': current_chunk['metadata']['sections'].copy(),
                            'has_table': True,
                            'has_image': False,
                            'images': [],
                            'tables': [block['metadata'].get('raw_table', '')]
                        }
                    }
                    current_words = len(table_text.split())
                else:
                    current_chunk['text'] += f" {table_text}"
                    current_chunk['metadata']['has_table'] = True
                    current_chunk['metadata']['tables'].append(block['metadata'].get('raw_table', ''))
                    current_words += len(table_text.split())
                continue
            
            # Handle paragraphs with sentence-based chunking
            if block['type'] == 'paragraph':
                sentences = sent_tokenize(block_text)
                
                for sentence in sentences:
                    sentence_words = len(sentence.split())
                    
                    # If adding this sentence exceeds chunk size, create new chunk
                    if current_words + sentence_words > self.chunk_size and current_words >= self.min_chunk_size:
                        # Add overlap from current chunk
                        chunks.append(current_chunk.copy())
                        
                        # Create overlap text
                        overlap_text = self._get_overlap_text(current_chunk['text'])
                        
                        current_chunk = {
                            'text': overlap_text + ' ' + sentence,
                            'metadata': {
                                'sections': current_chunk['metadata']['sections'].copy(),
                                'has_table': False,
                                'has_image': False,
                                'images': [],
                                'tables': []
                            }
                        }
                        current_words = len(current_chunk['text'].split())
                    else:
                        # Add sentence to current chunk
                        if current_chunk['text']:
                            current_chunk['text'] += ' ' + sentence
                        else:
                            current_chunk['text'] = sentence
                        
                        if block['metadata'].get('section') and block['metadata']['section'] not in current_chunk['metadata']['sections']:
                            current_chunk['metadata']['sections'].append(block['metadata']['section'])
                        
                        current_words += sentence_words
        
        # Add final chunk
        if current_chunk['text'].strip():
            chunks.append(current_chunk)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Extract overlap text from end of chunk"""
        words = text.split()
        if len(words) <= self.overlap:
            return text
        return ' '.join(words[-self.overlap:])
    
    def chunk_document(self, text: str) -> List[Dict[str, Any]]:
        """Main method to chunk a document"""
        blocks = self.parse_wikipedia_content(text)
        chunks = self.create_chunks(blocks)
        return chunks


if __name__ == "__main__":
    # Load raw corpus
    with open("data/raw_corpus.json") as f:
        corpus = json.load(f)
    
    chunker = SemanticChunker(
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE
    )
    
    chunked = []
    cid = 0
    
    # Chunk each document
    for doc in corpus["documents"]:
        chunks = chunker.chunk_document(doc["text"])
        
        for chunk in chunks:
            chunked.append({
                "chunk_id": cid,
                "url": doc["url"],
                "title": doc["title"],
                "text": chunk['text'],
                "metadata": {
                    "sections": chunk['metadata']['sections'],
                    "has_table": chunk['metadata']['has_table'],
                    "has_image": chunk['metadata']['has_image'],
                    "images": chunk['metadata']['images'],
                    "tables": chunk['metadata']['tables']
                }
            })
            cid += 1
    
    # Save chunked corpus
    with open("data/corpus_chunks.json", "w") as f:
        json.dump(chunked, f, indent=2)
    
    # Print statistics
    print(f"Total chunks created: {len(chunked)}")
    print(f"Chunks with tables: {sum(1 for c in chunked if c['metadata']['has_table'])}")
    print(f"Chunks with images: {sum(1 for c in chunked if c['metadata']['has_image'])}")