"""
Cloud Storage Module
====================
Integration with Azure Blob Storage and OneDrive (Microsoft Graph API).

Folder Structure: AccountName/LOB/PolicyNo-EffectiveDate/

Environment Variables:
    # Azure Blob Storage
    AZURE_STORAGE_CONNECTION_STRING
    AZURE_STORAGE_CONTAINER_NAME
    
    # OneDrive (Microsoft Graph)
    ONEDRIVE_CLIENT_ID
    ONEDRIVE_CLIENT_SECRET
    ONEDRIVE_TENANT_ID
    ONEDRIVE_DRIVE_ID (optional)
    ONEDRIVE_ROOT_FOLDER (optional)
    
    # Mode: azure|onedrive|both|mock
    STORAGE_MODE
"""

import os
import re
import tempfile
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from abc import ABC, abstractmethod
import dotenv

dotenv.load_dotenv()

# =============================================================================
# LOB Aliases
# =============================================================================

LOB_ALIASES = {
    "work": "wc", "workers": "wc", "workers comp": "wc",
    "vehicle": "auto", "car": "auto", "automobile": "auto",
    "house": "property", "home": "property", "fire": "property",
    "general": "gl", "liability": "gl", "general liability": "gl"
}

# =============================================================================
# Helper Functions
# =============================================================================

def _normalize_date(date_str: str) -> str:
    """Normalize date to folder format: DDMMYYYY"""
    if not date_str:
        return ""
    
    clean = re.sub(r'[-/.\s]', '', str(date_str))
    
    formats = [
        ("%d%m%Y", clean),
        ("%d-%m-%Y", date_str),
        ("%d/%m/%Y", date_str),
        ("%Y-%m-%d", date_str),
    ]
    
    for fmt, val in formats:
        try:
            return datetime.strptime(val, fmt).strftime("%d%m%Y")
        except ValueError:
            continue
    return clean


def _normalize_string(s: str) -> str:
    """Normalize string for matching."""
    return str(s).lower().strip() if s else ""


def _fuzzy_match(search: str, target: str) -> bool:
    """Flexible partial matching."""
    if not search:
        return True
    s = _normalize_string(search)
    t = _normalize_string(target)
    return s in t or t in s


def _parse_path(path: str) -> Optional[Dict[str, str]]:
    """Parse path: AccountName/LOB/PolicyNo-Date/filename.pdf"""
    parts = path.replace("\\", "/").strip("/").split("/")
    
    if len(parts) < 3:
        return None
    
    # Check if last part is a file
    if "." in parts[-1]:
        filename = parts[-1]
        folder = parts[-2] if len(parts) >= 2 else None
        lob = parts[-3] if len(parts) >= 3 else None
        account = parts[-4] if len(parts) >= 4 else None
    else:
        filename = None
        folder = parts[-1]
        lob = parts[-2] if len(parts) >= 2 else None
        account = parts[-3] if len(parts) >= 3 else None
    
    # Parse PolicyNo-Date
    policy = date = None
    if folder and "-" in folder:
        folder_parts = folder.split("-", 1)
        policy = folder_parts[0]
        date = folder_parts[1] if len(folder_parts) > 1 else None
    
    return {
        "account": account,
        "lob": lob,
        "policy_number": policy,
        "effective_date": date,
        "filename": filename
    }

# =============================================================================
# Abstract Storage Provider
# =============================================================================

class StorageProvider(ABC):
    """Base class for storage providers."""
    
    @abstractmethod
    def list_files(self, prefix: str = "") -> List[Dict[str, str]]:
        pass
    
    @abstractmethod
    def download_file(self, path: str) -> Tuple[bytes, str]:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        pass
    
    def download_to_temp(self, path: str) -> str:
        """Download to temp file and return path."""
        content, filename = self.download_file(path)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        with open(temp_path, 'wb') as f:
            f.write(content)
        return temp_path

# =============================================================================
# Azure Blob Storage
# =============================================================================

class AzureBlobStorage(StorageProvider):
    """Azure Blob Storage provider."""
    
    def __init__(self):
        self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "claims-documents")
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            from azure.storage.blob import BlobServiceClient
            self._client = BlobServiceClient.from_connection_string(self.connection_string)
        return self._client.get_container_client(self.container_name)
    
    def is_available(self) -> bool:
        return bool(self.connection_string)
    
    @property
    def source_name(self) -> str:
        return "Azure Blob Storage"
    
    def list_files(self, prefix: str = "") -> List[Dict[str, str]]:
        if not self.is_available():
            return []
        
        try:
            container = self._get_client()
            files = []
            
            for blob in container.list_blobs(name_starts_with=prefix or None):
                if blob.name.lower().endswith('.pdf'):
                    parsed = _parse_path(blob.name)
                    if parsed:
                        files.append({
                            "filename": parsed.get("filename") or os.path.basename(blob.name),
                            "folder_path": os.path.dirname(blob.name),
                            "full_path": blob.name,
                            "path": blob.name,
                            "source": self.source_name,
                            "account": parsed.get("account", ""),
                            "lob": parsed.get("lob", ""),
                            "policy_number": parsed.get("policy_number", ""),
                            "effective_date": parsed.get("effective_date", ""),
                        })
            return files
        except Exception:
            return []
    
    def download_file(self, path: str) -> Tuple[bytes, str]:
        container = self._get_client()
        blob = container.get_blob_client(path)
        return blob.download_blob().readall(), os.path.basename(path)

# =============================================================================
# OneDrive Storage
# =============================================================================

class OneDriveStorage(StorageProvider):
    """OneDrive provider via Microsoft Graph API."""
    
    def __init__(self):
        self.client_id = os.getenv("ONEDRIVE_CLIENT_ID")
        self.client_secret = os.getenv("ONEDRIVE_CLIENT_SECRET")
        self.tenant_id = os.getenv("ONEDRIVE_TENANT_ID")
        self.drive_id = os.getenv("ONEDRIVE_DRIVE_ID")
        self.root_folder = os.getenv("ONEDRIVE_ROOT_FOLDER", "Documents/Claims")
        self._token = None
        self._token_expiry = None
    
    def is_available(self) -> bool:
        return all([self.client_id, self.client_secret, self.tenant_id])
    
    @property
    def source_name(self) -> str:
        return "OneDrive"
    
    def _get_token(self) -> str:
        import requests
        from datetime import datetime, timedelta
        
        if self._token and self._token_expiry and datetime.now() < self._token_expiry:
            return self._token
        
        url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": "https://graph.microsoft.com/.default",
            "grant_type": "client_credentials"
        }
        
        resp = requests.post(url, data=data)
        resp.raise_for_status()
        token_data = resp.json()
        
        self._token = token_data["access_token"]
        self._token_expiry = datetime.now() + timedelta(seconds=token_data.get("expires_in", 3600) - 60)
        return self._token
    
    def _api_request(self, endpoint: str) -> dict:
        import requests
        headers = {"Authorization": f"Bearer {self._get_token()}"}
        url = f"https://graph.microsoft.com/v1.0{endpoint}"
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json() if resp.content else {}
    
    def list_files(self, prefix: str = "") -> List[Dict[str, str]]:
        if not self.is_available():
            return []
        
        try:
            files = []
            self._list_recursive(self.root_folder, files)
            return files
        except Exception:
            return []
    
    def _list_recursive(self, folder: str, files: List):
        try:
            drive = f"/drives/{self.drive_id}" if self.drive_id else "/me/drive"
            encoded = folder.replace(" ", "%20")
            endpoint = f"{drive}/root:/{encoded}:/children"
            
            resp = self._api_request(endpoint)
            
            for item in resp.get("value", []):
                if item.get("folder"):
                    self._list_recursive(f"{folder}/{item['name']}", files)
                elif item.get("file") and item["name"].lower().endswith(".pdf"):
                    path = f"{folder}/{item['name']}"
                    parsed = _parse_path(path)
                    
                    files.append({
                        "filename": item["name"],
                        "folder_path": folder,
                        "full_path": path,
                        "path": path,
                        "source": self.source_name,
                        "account": parsed.get("account", "") if parsed else "",
                        "lob": parsed.get("lob", "") if parsed else "",
                        "policy_number": parsed.get("policy_number", "") if parsed else "",
                        "effective_date": parsed.get("effective_date", "") if parsed else "",
                        "item_id": item["id"],
                    })
        except Exception:
            pass
    
    def download_file(self, path: str) -> Tuple[bytes, str]:
        import requests
        drive = f"/drives/{self.drive_id}" if self.drive_id else "/me/drive"
        encoded = path.replace(" ", "%20")
        endpoint = f"{drive}/root:/{encoded}:/content"
        
        headers = {"Authorization": f"Bearer {self._get_token()}"}
        url = f"https://graph.microsoft.com/v1.0{endpoint}"
        resp = requests.get(url, headers=headers, allow_redirects=True)
        resp.raise_for_status()
        
        return resp.content, os.path.basename(path)

# =============================================================================
# Storage Manager
# =============================================================================

class CloudStorageManager:
    """Unified manager for storage providers."""
    
    def __init__(self):
        self.providers: List[StorageProvider] = []
        self._init_providers()
    
    def _init_providers(self):
        mode = os.getenv("STORAGE_MODE", "mock").lower()
        
        if mode in ["azure", "both"]:
            azure = AzureBlobStorage()
            if azure.is_available():
                self.providers.append(azure)
        
        if mode in ["onedrive", "both"]:
            onedrive = OneDriveStorage()
            if onedrive.is_available():
                self.providers.append(onedrive)
    
    def search_files(
        self,
        account: Optional[str] = None,
        lob: Optional[str] = None,
        policy: Optional[str] = None,
        date: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Search across all providers with flexible matching."""
        results = []
        
        # Normalize search
        s_account = _normalize_string(account) if account else None
        s_lob = _normalize_string(lob) if lob else None
        s_policy = _normalize_string(policy) if policy else None
        s_date = _normalize_date(date) if date else None
        
        if s_lob and s_lob in LOB_ALIASES:
            s_lob = LOB_ALIASES[s_lob]
        
        for provider in self.providers:
            try:
                for f in provider.list_files():
                    if s_account and not _fuzzy_match(s_account, f.get("account", "")):
                        continue
                    if s_lob and s_lob != _normalize_string(f.get("lob", "")):
                        continue
                    if s_policy and not _fuzzy_match(s_policy, f.get("policy_number", "")):
                        continue
                    if s_date and s_date != _normalize_date(f.get("effective_date", "")):
                        continue
                    results.append(f)
            except Exception:
                continue
        
        return results
    
    def download_file(self, file_info: Dict[str, str]) -> str:
        """Download file and return local path."""
        source = file_info.get("source", "")
        path = file_info.get("path", file_info.get("full_path", ""))
        
        for provider in self.providers:
            if provider.source_name == source:
                return provider.download_to_temp(path)
        
        raise ValueError(f"Unknown source: {source}")

# =============================================================================
# Public API (drop-in replacement for mock_storage)
# =============================================================================

_manager: Optional[CloudStorageManager] = None


def _get_manager() -> CloudStorageManager:
    global _manager
    if _manager is None:
        _manager = CloudStorageManager()
    return _manager


def search_files(
    account_name: Optional[str] = None,
    lob: Optional[str] = None,
    policy_number: Optional[str] = None,
    date: Optional[str] = None,
    insured_name: Optional[str] = None,
    start_year: Optional[str] = None,
    end_year: Optional[str] = None
) -> List[Dict[str, str]]:
    """Search for files across cloud storage."""
    manager = _get_manager()
    
    # Fallback to mock storage if no cloud providers
    if not manager.providers:
        import mock_storage
        return mock_storage.search_files(
            account_name=account_name,
            lob=lob,
            policy_number=policy_number,
            date=date,
            insured_name=insured_name,
            start_year=start_year,
            end_year=end_year
        )
    
    return manager.search_files(account_name, lob, policy_number, date)


def download_file(file_info: Dict[str, str]) -> str:
    """Download file and return local path."""
    if file_info.get("source") == "Mock Storage":
        return file_info.get("path", file_info.get("full_path", ""))
    
    return _get_manager().download_file(file_info)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("☁️ Cloud Storage Test")
    print(f"Mode: {os.getenv('STORAGE_MODE', 'mock')}")
    
    results = search_files(account_name="chubbs", lob="auto")
    print(f"\nFound {len(results)} files")
    for r in results:
        print(f"  📄 {r.get('source')}: {r.get('filename')}")
