import urllib.request
import re

class WebTools:
    def fetch_url_content(self, url: str) -> str:
        """Fetches and extracts text content from a URL."""
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode('utf-8')
                
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator=' ', strip=True)
            except ImportError:
                # Fallback if bs4 is not installed
                text = re.sub(r'<[^>]+>', ' ', html)
                
            # Clean up multiple spaces
            text = re.sub(r'\s+', ' ', text).strip()
            # Truncate if extremely long (to save context window)
            if len(text) > 15000:
                text = text[:15000] + "\n... (Content truncated due to length)"
            return text
        except Exception as e:
            return f"Error fetching URL: {str(e)}"

    def browse_url(self, url: str) -> str:
        """Browser-like inspection: extracts text, links, and page structure."""
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode('utf-8')
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract basic info
            title = soup.title.string if soup.title else "No Title"
            
            # Extract links
            links = []
            for a in soup.find_all('a', href=True):
                links.append(f"[{a.text.strip()}]({a['href']})")
            
            # Extract main text
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text).strip()
            
            summary = [
                f"URL: {url}",
                f"Title: {title}",
                f"Links Found: {len(links)}",
                f"Text Content Sample: {text[:500]}...",
                "\nTop Links:",
                "\n".join(links[:10])
            ]
            return "\n".join(summary)
            
        except Exception as e:
            return f"Error browsing {url}: {str(e)}"
