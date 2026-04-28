"""
api_client.py
=============
Cliente HTTP para integração com a API de vídeos do projeto
(endpoints /movies/import e /videos/processed).

Permite buscar vídeos processados e alimentar o coletor de métricas
automaticamente, fechando o loop do pipeline MLOps.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class VideoRecord:
    id: str
    status: str                     # 'Enviado' | 'Em processamento' | 'Processado' | 'Erro'
    bucket: str
    key: str
    processed_key: Optional[str] = None
    processed_at: Optional[str] = None
    size: Optional[int] = None
    processed_size: Optional[int] = None
    content_type: Optional[str] = None
    last_error: Optional[str] = None


class VideoAPIClient:
    """
    Cliente para a API de vídeos do sistema de detecção.

    Parâmetros
    ----------
    base_url : str
        URL base da API, ex.: "http://localhost:3000"
    token : str, opcional
        JWT token para endpoints autenticados.
    timeout : int
        Timeout das requisições em segundos.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        token: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    # ── helpers HTTP ──────────────────────────────────────────────────────────

    def _headers(self, auth: bool = False) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if auth and self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _get(self, path: str, auth: bool = False) -> Any:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, headers=self._headers(auth))
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} em GET {url}: {exc.read().decode()}") from exc

    def _post_json(self, path: str, payload: dict, auth: bool = False) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=self._headers(auth), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} em POST {url}: {exc.read().decode()}") from exc

    # ── endpoints ─────────────────────────────────────────────────────────────

    def list_processed_videos(self) -> List[VideoRecord]:
        """
        Retorna todos os vídeos com status 'Processado'.
        Usa GET /movies (filtragem feita no cliente).
        """
        raw = self._get("/movies", auth=True)
        records = raw if isinstance(raw, list) else raw.get("data", [])
        return [
            VideoRecord(
                id=r.get("id", ""),
                status=r.get("status", ""),
                bucket=r.get("bucket", ""),
                key=r.get("key", ""),
                processed_key=r.get("processedKey"),
                processed_at=r.get("processedAt"),
                size=r.get("size"),
                processed_size=r.get("processedSize"),
                content_type=r.get("contentType"),
                last_error=r.get("lastError"),
            )
            for r in records
            if r.get("status") == "Processado"
        ]

    def get_video(self, video_id: str) -> VideoRecord:
        """Retorna um vídeo específico pelo UUID."""
        r = self._get(f"/movies/{video_id}", auth=True)
        return VideoRecord(
            id=r.get("id", video_id),
            status=r.get("status", ""),
            bucket=r.get("bucket", ""),
            key=r.get("key", ""),
            processed_key=r.get("processedKey"),
            processed_at=r.get("processedAt"),
            size=r.get("size"),
            processed_size=r.get("processedSize"),
            content_type=r.get("contentType"),
            last_error=r.get("lastError"),
        )

    def notify_processed(
        self,
        video_id: str,
        processed_object_key: str,
        processed_size: int,
    ) -> Dict:
        """
        Notifica a API que um vídeo foi processado diretamente no MinIO.
        Corresponde ao endpoint POST /videos/{id}/processed/notify.
        """
        return self._post_json(
            f"/videos/{video_id}/processed/notify",
            payload={
                "processedObjectKey": processed_object_key,
                "processedSize": processed_size,
            },
            auth=True,
        )

    def pipeline_stats(self) -> Dict:
        """
        Retorna estatísticas gerais do pipeline:
        total de vídeos por status.
        """
        try:
            raw = self._get("/movies", auth=True)
            records = raw if isinstance(raw, list) else raw.get("data", [])
        except RuntimeError:
            return {"error": "Não foi possível conectar à API"}

        stats: Dict[str, int] = {}
        for r in records:
            s = r.get("status", "Desconhecido")
            stats[s] = stats.get(s, 0) + 1
        stats["total"] = len(records)
        return stats
