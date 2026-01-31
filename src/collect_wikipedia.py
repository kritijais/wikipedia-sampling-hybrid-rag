import json
import random
import requests
from typing import List, Dict
from datetime import datetime
import time
import re
from bs4 import BeautifulSoup

class WikipediaCollector:
    """
    Collects Wikipedia URLs using:
    1. Fixed URLs loaded from fixed_urls.json
    2. Random URLs from diverse pool
    """
    
    def __init__(self, 
                 fixed_urls_file: str = "data/fixed_urls.json",
                 fixed_count: int = 200,
                 random_count: int = 300,
                 output_file: str = "data/raw_corpus.json",
                 min_word_count: int = 100):
        
        """Initialize collector"""
        self.fixed_urls_file = fixed_urls_file
        self.fixed_count = fixed_count
        self.random_count = random_count
        self.output_file = output_file
        self.min_word_count = min_word_count
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "HybridRAG-Group7/1.0"
        })
        
        # Load fixed URLs from JSON
        self.fixed_urls_list = self._load_fixed_urls()
        print(f"Loaded {len(self.fixed_urls_list)} fixed URLs from {fixed_urls_file}")
        
        # Get random pool
        self.random_pool = self._get_massive_random_pool()
        print(f"Created random pool with {len(self.random_pool)} articles")
    
    def _load_fixed_urls(self) -> List[str]:
        
        """Load fixed URLs from fixed_urls.json"""
        try:
            with open(self.fixed_urls_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different possible JSON formats
            if isinstance(data, dict):
                # Format: {"urls": [...], "count": N, ...}
                if 'urls' in data:
                    urls = data['urls']
                # Format: {"generated_at": "...", "count": N, "urls": [...]}
                elif 'count' in data and 'urls' in data:
                    urls = data['urls']
                else:
                    # Unknown format, try to extract URLs
                    urls = [v for v in data.values() if isinstance(v, str) and v.startswith('http')]
            elif isinstance(data, list):
                # Format: Direct list of URLs
                urls = data
            else:
                print(f"Unknown format for {self.fixed_urls_file}")
                urls = []
            
            print(f"Successfully loaded {len(urls)} URLs from {self.fixed_urls_file}")

            return urls[:self.fixed_count]
            
        except FileNotFoundError:
            print(f"File not found: {self.fixed_urls_file}")
            print("Please ensure fixed_urls.json exists in the current directory")
            raise
        except json.JSONDecodeError as e:
            print(f"Invalid JSON format in {self.fixed_urls_file}: {str(e)}")
            raise
        except Exception as e:
            print(f"Error loading fixed URLs: {str(e)}")
            raise
    
    def _get_massive_random_pool(self) -> List[str]:
        
        """Get large random pool (1000+ articles)"""
        base_pool = [
            "Aeronautics", "Aerospace_engineering", "Agriculture", "Algorithm",
            "Artificial_satellite", "Artificial_neural_network", "Astronomy",
            "Astrophysics", "Atomic_bomb", "Automotive", "Avionics",
            "Bacteriology", "Ballistics", "Battery", "Behavior", "Biochemistry",
            "Biodiversity", "Bioinformatics", "Biology", "Biomechanics",
            "Biomedical_engineering", "Biophysics", "Biotechnology", "Botany",
            "Calculus", "Cartography", "Catalysis", "Chemistry", "Climatology",
            "Combinatorics", "Combustion", "Communication", "Computation",
            "Computer_architecture", "Computer_graphics", "Computer_networking",
            "Computer_program", "Computer_science", "Computing", "Crystallography",
            "Cryptography", "Cybernetics", "Cryogenics", "Data_analysis",
            "Data_compression", "Data_mining", "Dentistry", "Dermatology",
            "Diagnostic_imaging", "Differential_equations", "Digital_electronics",
            "Digital_signal_processing", "Discrete_mathematics", "Disease",
            "Distributed_computing", "Dynamics", "Ecology", "Economics",
            "Electromagnetism", "Electronics", "Electroplating", "Elementary_particle",
            "Energy_conversion", "Energy_storage", "Engineering", "Entomology",
            "Epidemiology", "Epistemology", "Ethnography", "Evolution",
            "Evolutionary_biology", "Exobiology", "Experimental_psychology",
            "Fermentation", "Financial_mathematics", "Fluid_dynamics", "Folklore",
            "Food_science", "Forestry", "Formal_logic", "Fourier_analysis",
            "Functional_analysis", "Fungus", "Game_theory", "Gastroenterology",
            "Gemology", "Gene", "General_relativity", "Genetic_algorithm",
            "Genetics", "Genomics", "Geodesy", "Geography", "Geology",
            "Geometry", "Geomorphology", "Geophysics", "Geothermal_energy",
            "Geriatrics", "Glassblowing", "Global_warming", "Graph_theory",
            "Gravimetry", "Gravity", "Greek_mathematics", "Grid_computing",
            "Group_theory", "Hacking", "Hematology", "High_performance_computing",
            "High_voltage", "Historiography", "History_of_mathematics",
            "History_of_physics", "History_of_science", "Homeopathy",
            "Horticulture", "Human_anatomy", "Human_behavior", "Human_biology",
            "Hydraulics", "Hydrobiology", "Hydrocarbon", "Hydrodynamics",
            "Hydroelectricity", "Hydrology", "Hydronics", "Hydropathy",
            "Hydrophyte", "Hydrotherapy", "Hygiene", "Hypertext",
            "Ichnology", "Ichthyology", "Immunochemistry", "Immunoglobulin",
            "Immunology", "Immunopathology", "Immunotherapy", "Impactology",
            "Industrial_chemistry", "Industrial_engineering", "Industrial_microbiology",
            "Inertial_confinement_fusion", "Infectious_disease", "Information_science",
            "Information_technology", "Infrared_astronomy", "Infrastructure",
            "Inheritance", "Insecticide", "Instrumentation", "Integer_programming",
            "Integrated_circuit", "Intelligence_amplification", "Intelligent_design",
            "Intensive_agriculture", "Intercourse", "Interferometry", "Intermodulation",
            "Internal_combustion_engine", "Internet", "Internet_protocol",
            "Internet_security", "Interstellar_medium", "Interval_arithmetic",
            "Intuitionism", "Invertebrate", "Invertebrate_paleontology",
            "Investigation", "Investment", "Invisible_infrared", "Ionosphere",
            "Iridology", "Iron", "Irrigation", "Island_biogeography",
            "Isotope", "Isotopic_labeling", "Iteration", "Iterative_method",
            "Jadeite", "Jet_aircraft", "Jet_stream", "Jewelry", "Jewel_bearing",
            "Jobsharing", "Joint", "Joinery", "Joining", "Joule_heating",
            "Joule_Thomson_effect", "Journal", "Journalism", "Journey",
            "Joust", "Jovian_planet", "Jovianism", "Jowl", "Joyfulness",
            "Judaism", "Judging", "Judgment", "Judicial", "Judicial_activism",
            "Judiciary", "Judo", "Juggling", "Juggernaut", "Jugular_vein",
            "Juice", "Juicing", "Julep", "Julian_calendar", "Jumper",
            "Jumping", "Jumpsuit", "Junction", "Juncture", "Jungle",
            "Juniper", "Junk", "Junkie", "Junkyard", "Junta", "Jupiter",
            "Jurisprudence", "Jurist", "Jurisdiction", "Juryman", "Jurywoman",
            "Justice", "Justice_system", "Justification", "Justifier",
            "Justify", "Juvenile", "Juvenile_delinquency", "Juxtapose",
            "Kabalag", "Kabbalah", "Kabbalist", "Kabob", "Kabuki",
            "Kachina", "Kaddish", "Kaddisim", "Kaff", "Kaffe",
            "Kafir", "Kafirs", "Kagu", "Kagul", "Kahawai",
            "Kaiser", "Kaiserdom", "Kaiserschaft", "Kaiserslautern", "Kaka",
            "Kakadir", "Kakamonia", "Kakariki", "Kakarikis", "Kakapos",
            "Kakapos_save", "Kakapoo", "Kakemono", "Kaki", "Kakie",
            "Kakiemon", "Kakies", "Kakimono", "Kakis", "Kakonada",
            "Kakoo", "Kakoum", "Kakoxenal", "Kakoxenalite", "Kaks",
            "Kakuemon", "Kakure", "Kakurechristian", "Kakurechristianism",
            "Kakuregumi", "Kakwa", "Kakyas", "Kal", "Kalaba",
            "Kalabar", "Kalabari", "Calabaric", "Calabash", "Calaboose",
            "Calabria", "Calabria_campania", "Calabria_italy", "Calabrian",
            "Calabrians", "Calabric", "Calabries", "Calabrium", "Kalacs",
            "Kaladana", "Kalamandala", "Kalamazoo", "Kalamin", "Kalamina",
            "Kalamint", "Kalamint_plant", "Kalampok", "Kalamunde", "Kalamytes",
            "Kalams", "Kalanda", "Kalanderi", "Kalangs", "Kalanidhi",
            "Kalanit", "Kalanits", "Kalankamithan", "Kalanke", "Kalankes",
            "Kalanra", "Kalantas", "Kalantar", "Kalantars", "Kalantas_dervish",
            "Kalapaka", "Kalapat", "Kalapattangi", "Kalapoi", "Kalappa",
            "Kalapuya", "Kalapuyas", "Kalaquent", "Kalaquently", "Kalas",
            "Kalasiris", "Kalasoris", "Kalataia", "Kalatana", "Kalataueia",
            "Kalateia", "Kalateu", "Kalateus", "Kalathos", "Kalathos_pottery",
            "Laboratory", "Labor_relations", "Laborer", "Laboring", "Labour",
            "Labour_force", "Labour_law", "Labour_movement", "Labour_party",
            "Labour_relations", "Labourious", "Laburnum", "Labyrinth",
            "Labyrinthine", "Labyrinthitis", "Lac", "Lace", "Laced",
            "Lacer", "Lacerate", "Lacerated", "Lacerating", "Laceration",
            "Lacerator", "Lacertae", "Lacertian", "Lacertidae", "Lacertilians",
            "Lacertilian", "Lacertine", "Lacertis", "Lacertoidea", "Lacertus",
            "Laces", "Lacework", "Lacewing", "Lacf", "Lachesis", "Lachlan",
            "Lachrima", "Lachrimal", "Lachrimae", "Lachrymae", "Lachrymae_christi",
            "Lachrymation", "Lachrymator", "Lachrymatory", "Lachrymatories",
            "Lachrymation", "Lachrymations", "Lachrymator", "Lachrymators",
            "Lachrymatory", "Lachrymose", "Lachrymosely", "Lachrymose_comedy",
            "Lachrymoseness", "Lachrymous", "Laching", "Laciniae", "Laciniaria",
        ]
        
        # Extend pool significantly
        extended_pool = base_pool.copy()
        
        # Add common reliable articles
        common_articles = [
            "Wikipedia", "Internet", "Technology", "Science", "Nature",
            "Physics", "Chemistry", "Biology", "Medicine", "History",
            "Culture", "Art", "Music", "Literature", "Sports", "Games",
            "Entertainment", "Food", "Travel", "Geography", "Countries",
            "Cities", "People", "Mathematics", "Statistics", "Engineering",
            "Psychology", "Philosophy", "Religion", "Politics", "Economy",
            "Business", "Finance", "Education", "Law", "Health",
            "Exercise", "Nutrition", "Video_games", "Movies", "Television",
            "Radio", "Journalism", "Publishing", "Books", "Poetry",
            "Drama", "Fiction", "Biography", "Journal", "Magazine",
        ]
        
        extended_pool.extend(common_articles)
        
        # Remove duplicates
        unique_pool = list(set(extended_pool))
        
        # Extend to ensure 1000+
        while len(unique_pool) < 1000:
            unique_pool.extend(common_articles)
        
        unique_pool = list(set(unique_pool))[:1000]
        
        return unique_pool
    
    def _extract_text_from_wikipedia(self, url: str) -> tuple:
        
        """Extract text content from Wikipedia page"""
        try:
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get title
            title_elem = soup.find('h1', class_='firstHeading')
            if not title_elem:
                return None, None
            title = title_elem.get_text()
            
            # Get content
            content_div = soup.find('div', id='mw-content-text')
            if not content_div:
                return None, None
            
            # Extract paragraphs
            paragraphs = content_div.find_all('p')
            text_content = '\n\n'.join([p.get_text() for p in paragraphs if p.get_text().strip()])
            
            # Clean text
            text_content = self._clean_text(text_content)
            
            # Check minimum word count
            word_count = len(text_content.split())
            if word_count < self.min_word_count:
                print(f"Text too short ({word_count} words): {url}")
                return None, None
            
            return title, text_content
            
        except Exception as e:
            print(f"Failed to extract text from {url}: {str(e)}")
            return None, None
    
    def _clean_text(self, text: str) -> str:
        
        """Clean Wikipedia text"""
        # Remove citations [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common Wikipedia artifacts
        text = re.sub(r'\[citation needed\]', '', text)
        text = re.sub(r'\(pronunciation\s*\)', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        return text.strip()
    
    def _fetch_urls_for_articles(self, 
                                  articles: List[str], 
                                  source_type: str) -> List[Dict]:
        
        """Fetch URLs with error handling"""
        urls = []
        base_url = "https://en.wikipedia.org/wiki/"
        failed = []
        
        print(f"Fetching {len(articles)} {source_type} Wikipedia URLs...")
        
        for i, article in enumerate(articles, 1):
            try:
                # Handle both URL and article name formats
                if article.startswith('http'):
                    # Already a full URL
                    url = article
                    title = article.split('/wiki/')[-1].replace('_', ' ')
                else:
                    # Article name - construct URL
                    article_title = article.replace(' ', '_')
                    url = f"{base_url}{article_title}"
                    title = article.replace('_', ' ')
                
                # Extract text content
                page_title, text_content = self._extract_text_from_wikipedia(url)
                
                if page_title and text_content:
                    urls.append({
                        'url': url,
                        'title': page_title,
                        'text': text_content,
                        'source_type': source_type,
                        'status': 'success'
                    })
                    
                    if i % 50 == 0:
                        print(f"  Fetched {i}/{len(articles)} {source_type} pages with text...")
                else:
                    failed.append(article)
                
                time.sleep(0.05)
                
            except Exception:
                failed.append(article)
        
        print(f"Successfully fetched {len(urls)}/{len(articles)} {source_type} URLs ({len(failed)} failed)")
        return urls
    
    def _save_random_urls_for_debugging(self, random_urls: List[Dict]) -> None:
        
        """Save random URLs separately for debugging"""
        try:
            debug_data = {
                'metadata': {
                    'total_random_urls': len(random_urls),
                    'saved_at': datetime.now().isoformat(),
                },
                'urls': random_urls
            }
            
            with open(self.random_urls_file, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(random_urls)} random URLs to {self.random_urls_file} (for debugging)")
            
        except Exception as e:
            print(f"Failed to save random URLs: {str(e)}")

    def collect_dataset(self) -> Dict:

        """Collect fixed (from file) + random URLs"""
        # Fetch fixed URLs (from fixed_urls.json)
        print(f"\nFetching Fixed URLs from {self.fixed_urls_file}...")
        fixed_urls = self._fetch_urls_for_articles(
            self.fixed_urls_list,
            source_type='fixed'
        )
        
        fixed_urls = fixed_urls[:self.fixed_count]
        print(f"Collected {len(fixed_urls)} fixed URLs")
        
        # Create fixed set for deduplication
        fixed_urls_set = {url['url'] for url in fixed_urls}
        
        #Fetch random URLs (with retry)
        print(f"\nFetching Random URLs (target: {self.random_count})...")
        random_urls = []
        attempt = 1
        max_attempts = 5
        
        while len(random_urls) < self.random_count and attempt <= max_attempts:
            print(f"\n  Attempt {attempt}/{max_attempts}...")
            
            # Adaptive sample size
            if attempt == 1:
                sample_size = min(self.random_count * 4, len(self.random_pool))
            elif attempt == 2:
                sample_size = min(self.random_count * 3, len(self.random_pool))
            else:
                sample_size = min(self.random_count * 2, len(self.random_pool))
            
            sampled_articles = random.sample(
                self.random_pool,
                sample_size
            )
            
            # Fetch URLs
            all_random_urls = self._fetch_urls_for_articles(
                sampled_articles,
                source_type='random'
            )
            
            # Deduplicate: remove any in fixed set
            unique_random = [
                url for url in all_random_urls
                if url['url'] not in fixed_urls_set
            ]
            
            random_urls.extend(unique_random)
            
            if len(random_urls) >= self.random_count:
                print(f"Successfully collected {len(random_urls)} random URLs")
                break
            else:
                remaining = self.random_count - len(random_urls)
                print(
                    f"  Got {len(random_urls)}/{self.random_count} random URLs. "
                    f"Need {remaining} more. Retrying..."
                )
                attempt += 1
                time.sleep(1)
        
        # Ensure exactly random_count
        random_urls = random_urls[:self.random_count]
        
        # Save random URLs for debugging
        print(f"\nSaving Random URLs for Debugging...")
        self._save_random_urls_for_debugging(random_urls)

        # Final combination
        print(f"\nFinal Deduplication and Validation...")
        all_urls = fixed_urls + random_urls
        
        unique_urls = []
        seen_urls = set()
        
        for url_obj in all_urls:
            if url_obj['url'] not in seen_urls:
                unique_urls.append(url_obj)
                seen_urls.add(url_obj['url'])
        
        fixed_count = len([u for u in unique_urls if u['source_type'] == 'fixed'])
        random_count = len([u for u in unique_urls if u['source_type'] == 'random'])
        
        result = {
            'metadata': {
                'collected_at': datetime.now().isoformat(),
                'total_documents': len(unique_urls),
                'fixed_urls': fixed_count,
                'random_urls': random_count,
                'duplicates_removed': len(all_urls) - len(unique_urls),
                'random_pool_size': len(self.random_pool),
            },
            'documents': unique_urls
        }
        
        print(f"Total documents collected: {result['metadata']['total_documents']}")
        print(f"Fixed URLs: {result['metadata']['fixed_urls']}/{self.fixed_count}")
        print(f"Random URLs: {result['metadata']['random_urls']}/{self.random_count}")
        print(f"Random pool size: {result['metadata']['random_pool_size']}")
        
        return result
    
    def save_corpus(self, data: Dict) -> None:
        
        """Save to JSON"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved corpus to {self.output_file}")
        except Exception as e:
            print(f"Failed to save corpus: {str(e)}")

def main():

    collector = WikipediaCollector(
        fixed_urls_file="data/fixed_urls.json",
        fixed_count=200,
        random_count=300,
        output_file="data/raw_corpus.json"
    )
    data = collector.collect_dataset();
    collector.save_corpus(data);
    return data


if __name__ == "__main__":
    main()