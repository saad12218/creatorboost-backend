import base64
import asyncio
import html
import io
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict
from functools import lru_cache
from urllib.parse import parse_qs, urlparse

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from PIL import Image, ImageOps
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl, ValidationError, field_validator

load_dotenv()

OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_IMAGE_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "low")
OPENAI_IMAGE_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1024x1024")
FRONTEND_URL = os.getenv(
	"FRONTEND_URL", "http://localhost:3000,http://localhost:3002"
)
YOUTUBE_OEMBED_URL = "https://www.youtube.com/oembed"
RATE_LIMIT_WINDOW_SECONDS = 60
ANALYZE_RATE_LIMIT = int(os.getenv("ANALYZE_RATE_LIMIT_PER_MINUTE", "10"))
THUMBNAIL_RATE_LIMIT = int(os.getenv("THUMBNAIL_RATE_LIMIT_PER_MINUTE", "5"))
logger = logging.getLogger("uvicorn.error")
thumbnail_reference_lock = threading.Lock()
rate_limit_lock = threading.Lock()
rate_limit_hits: dict[tuple[str, str], list[float]] = defaultdict(list)


def client_ip(request: Request) -> str:
	forwarded = request.headers.get("x-forwarded-for")
	if forwarded:
		return forwarded.split(",")[0].strip()
	return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request, bucket: str, limit: int) -> None:
	key = (bucket, client_ip(request))
	now = time.monotonic()
	with rate_limit_lock:
		hits = rate_limit_hits[key]
		cutoff = now - RATE_LIMIT_WINDOW_SECONDS
		while hits and hits[0] < cutoff:
			hits.pop(0)
		if len(hits) >= limit:
			raise HTTPException(status_code=429, detail="Too many requests. Please slow down and try again shortly.")
		hits.append(now)


class AnalyzeRequest(BaseModel):
	youtube_url: HttpUrl

	@field_validator("youtube_url")
	@classmethod
	def validate_youtube_url(cls, value: HttpUrl) -> HttpUrl:
		if extract_video_id(str(value)) is None:
			raise ValueError("Provide a valid YouTube video URL.")
		return value


class VideoInfo(BaseModel):
	id: str
	title: str
	author: str
	thumbnail: HttpUrl
	description: str | None = None
	transcript: str | None = None
	context_source: str


class ThumbnailRequest(AnalyzeRequest):
	concept: str = Field(min_length=1, max_length=2000)
	video: VideoInfo | None = None


class TitleSuggestion(BaseModel):
	title: str
	score: int = Field(ge=0, le=100)
	reason: str


class ThumbnailIdea(BaseModel):
	concept: str
	short_text: str = Field(min_length=1, max_length=80)
	composition: str
	subject: str
	background: str
	lighting: str
	graphic_elements: str
	click_reason: str

	@property
	def text(self) -> str:
		return self.short_text

	@property
	def visual(self) -> str:
		return " ".join(
			(
				self.composition,
				self.subject,
				self.background,
				self.lighting,
				self.graphic_elements,
				self.click_reason,
			)
		)


class ContentPackage(BaseModel):
	titles: list[TitleSuggestion]
	description: str
	keywords: list[str]
	hashtags: list[str]
	seo_score: int = Field(ge=0, le=100)
	seo_summary: str
	thumbnail_ideas: list[ThumbnailIdea]

	@field_validator("titles")
	@classmethod
	def validate_title_count(cls, value: list[TitleSuggestion]) -> list[TitleSuggestion]:
		if len(value) != 10:
			raise ValueError("The AI response must contain exactly 10 titles.")
		return value

	@field_validator("keywords")
	@classmethod
	def validate_keyword_count(cls, value: list[str]) -> list[str]:
		if not 10 <= len(value) <= 15:
			raise ValueError("The AI response must contain 10 to 15 keywords.")
		return value

	@field_validator("hashtags")
	@classmethod
	def validate_hashtag_count(cls, value: list[str]) -> list[str]:
		if not 8 <= len(value) <= 12:
			raise ValueError("The AI response must contain 8 to 12 hashtags.")
		return value

	@field_validator("thumbnail_ideas")
	@classmethod
	def validate_thumbnail_count(cls, value: list[ThumbnailIdea]) -> list[ThumbnailIdea]:
		if len(value) != 3:
			raise ValueError("The AI response must contain exactly 3 thumbnail concepts.")
		return value


class AnalyzeResponse(BaseModel):
	video: VideoInfo
	result: ContentPackage


class ThumbnailResponse(BaseModel):
	image_url: str


app = FastAPI(title="CreatorBoost API", version="1.0.0")

allowed_origins = [origin.strip() for origin in FRONTEND_URL.split(",") if origin.strip()]
app.add_middleware(
	CORSMiddleware,
	allow_origins=allowed_origins,
	allow_credentials=True,
	allow_methods=["GET", "POST"],
	allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
	request: Request, exception: RequestValidationError
) -> JSONResponse:
	del request
	return JSONResponse(
		status_code=400,
		content={"detail": "Invalid request: " + "; ".join(error["msg"] for error in exception.errors())},
	)


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")
	return OpenAI(api_key=api_key)


def extract_video_id(youtube_url: str) -> str | None:
	parsed = urlparse(youtube_url)
	hostname = parsed.netloc.lower().split(":")[0]
	if hostname == "youtu.be":
		candidate = parsed.path.strip("/").split("/")[0]
	elif hostname in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
		if parsed.path == "/watch":
			candidate = parse_qs(parsed.query).get("v", [""])[0]
		elif parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
			candidate = parsed.path.split("/")[2]
		else:
			return None
	else:
		return None

	if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
		return candidate
	return None


def get_meta_content(page: str, property_name: str) -> str | None:
	pattern = re.compile(
		rf'<meta[^>]+(?:property|name)=["\']{re.escape(property_name)}["\'][^>]+content=["\']([^"\']*)["\']',
		re.IGNORECASE,
	)
	match = pattern.search(page)
	if not match:
		return None
	return html.unescape(match.group(1)).strip() or None


def get_optional_transcript(video_id: str) -> str | None:
	try:
		from youtube_transcript_api import YouTubeTranscriptApi

		transcript = YouTubeTranscriptApi().fetch(video_id)
		text = " ".join(item.text for item in transcript)
		return text[:12000] or None
	except Exception:
		return None


def get_video_metadata(youtube_url: str, video_id: str) -> VideoInfo:
	try:
		oembed_response = requests.get(
			YOUTUBE_OEMBED_URL,
			params={"url": youtube_url, "format": "json"},
			timeout=10,
		)
		oembed_response.raise_for_status()
		metadata = oembed_response.json()
		description = None
		try:
			page_response = requests.get(
				f"https://www.youtube.com/watch?v={video_id}",
				headers={"User-Agent": "Mozilla/5.0 CreatorBoost/1.0"},
				timeout=10,
			)
			page_response.raise_for_status()
			page = page_response.text
			description = get_meta_content(page, "og:description") or get_meta_content(page, "description")
		except requests.RequestException:
			pass
		transcript = get_optional_transcript(video_id)
		context_source = "metadata and transcript" if transcript else "metadata only"
		return VideoInfo(
			id=video_id,
			title=metadata["title"],
			author=metadata.get("author_name", "Unknown creator"),
			thumbnail=metadata.get(
				"thumbnail_url", f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
			),
			description=description,
			transcript=transcript,
			context_source=context_source,
		)
	except (requests.RequestException, KeyError, TypeError, ValueError) as error:
		raise HTTPException(
			status_code=502,
			detail="The submitted YouTube video is inaccessible or its public metadata is unavailable.",
		) from error


def generate_content(video: VideoInfo) -> ContentPackage:
	client = get_openai_client()
	instructions = """
You are an expert YouTube content strategist. Analyze only the submitted video's context. Do not
infer a different topic, invent facts, or use a generic demo topic. If transcript context is absent,
say so in seo_summary and base the package on the available title, channel, and description only.
Return only valid JSON matching this schema:
{
  "titles": [{"title": "string", "score": 0, "reason": "string"}],
  "description": "string",
  "keywords": ["string"],
  "hashtags": ["string"],
  "seo_score": 0,
  "seo_summary": "string",
	"thumbnail_ideas": [{
		"concept": "string",
		"short_text": "2 to 5 words, maximum 30 characters",
		"composition": "string",
		"subject": "string",
		"background": "string",
		"lighting": "string",
		"graphic_elements": "string",
		"click_reason": "string"
	}]
}

Create exactly 10 titles, 10-15 keywords, 8-12 hashtags, and exactly 3 thumbnail ideas.
Titles must be compelling and specific to this video without misleading clickbait. Thumbnail text must be short.
The description should naturally use relevant keywords without keyword stuffing.
Each thumbnail idea must be meaningfully different and must include a clear composition,
subject/object placement, background, lighting/style, short text, graphic elements, and why
the concept could attract clicks. Ground every field in the submitted video's title, description,
transcript, channel, and available public thumbnail context. Never use a generic example topic.
seo_score is only an estimated AI assessment,
never an official YouTube score or ranking guarantee.
"""
	user_input = json.dumps(
		{
			"video_id": video.id,
			"title": video.title,
			"channel": video.author,
			"description": video.description,
			"transcript": video.transcript,
			"context_source": video.context_source,
		},
		ensure_ascii=False,
	)
	try:
		response = client.responses.create(
			model=OPENAI_TEXT_MODEL,
			instructions=instructions,
			input=f"Create the publishing package as json for this exact public YouTube video context: {user_input}",
			text={"format": {"type": "json_object"}},
		)
		raw_output = response.output_text.strip()
		result = json.loads(raw_output)
		return ContentPackage.model_validate(result)
	except json.JSONDecodeError as error:
		raise HTTPException(status_code=502, detail="OpenAI returned invalid JSON.") from error
	except ValidationError as error:
		raise HTTPException(
			status_code=502,
			detail=f"OpenAI returned an invalid content package: {error.errors()[0]['msg']}",
		) from error
	except Exception as error:
		raise HTTPException(status_code=502, detail="OpenAI content generation failed.") from error


@lru_cache(maxsize=64)
def get_thumbnail_reference_bytes(thumbnail_url: str) -> bytes | None:
	with thumbnail_reference_lock:
		return _download_thumbnail_reference_bytes(thumbnail_url)


@lru_cache(maxsize=64)
def _download_thumbnail_reference_bytes(thumbnail_url: str) -> bytes | None:
	try:
		response = requests.get(thumbnail_url, timeout=10)
		response.raise_for_status()
		return response.content
	except (requests.RequestException, OSError, ValueError):
		return None


def get_thumbnail_reference(video: VideoInfo) -> io.BytesIO | None:
	content = get_thumbnail_reference_bytes(str(video.thumbnail))
	if content is None:
		return None
	output = io.BytesIO(content)
	output.name = f"youtube-{video.id}.jpg"
	return output


def generate_thumbnail(video: VideoInfo, concept: str) -> str:
	started_at = time.perf_counter()
	client = get_openai_client()
	prompt = (
		"Design a finished, upload-ready 16:9 YouTube thumbnail for the exact video context below. "
		"This is a high-CTR creator thumbnail, not a poster, slide, illustration, stock photo, or collage. "
		"Use the supplied reference image only to understand the real subject, people, setting, colors, and visual story; "
		"recompose it into a bold original thumbnail rather than copying its layout. "
		"Make the main subject large and instantly readable on a phone, with a clear focal point, depth, thick subject separation, "
		"dramatic but believable lighting, strong contrast, professional color grading, and minimal clutter. "
		"Render the selected short text as large, bold, legible thumbnail typography. Use arrows, circles, highlights, glow, "
		"or an emoji only when the selected concept calls for it; never add decorative graphics randomly. "
		"Do not add logos, watermarks, tiny text, unrelated objects, or facts absent from the video context.\n\n"
		f"Video title: {video.title}\n"
		f"Channel: {video.author}\n"
		f"Description: {video.description or 'Not available'}\n"
		f"Transcript context: {video.transcript or 'Not available; use metadata only'}\n"
		f"Selected thumbnail concept: {concept}"
	)
	try:
		reference_started_at = time.perf_counter()
		reference = get_thumbnail_reference(video)
		reference_ms = (time.perf_counter() - reference_started_at) * 1000
		openai_started_at = time.perf_counter()
		if reference is not None:
			response = client.images.edit(
				model=OPENAI_IMAGE_MODEL,
				image=reference,
				prompt=prompt,
				size=OPENAI_IMAGE_SIZE,
				quality=OPENAI_IMAGE_QUALITY,
			)
		else:
			response = client.images.generate(
				model=OPENAI_IMAGE_MODEL,
				prompt=prompt,
				size=OPENAI_IMAGE_SIZE,
				quality=OPENAI_IMAGE_QUALITY,
			)
		openai_ms = (time.perf_counter() - openai_started_at) * 1000
		image_data = response.data[0].b64_json if response.data else None
		if not image_data:
			raise HTTPException(status_code=502, detail="OpenAI returned no thumbnail image data.")
		image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
		preview_size = (1280, 720)
		fitted = ImageOps.contain(image, preview_size)
		canvas = Image.new("RGB", preview_size, (23, 27, 32))
		canvas.paste(fitted, ((preview_size[0] - fitted.width) // 2, (preview_size[1] - fitted.height) // 2))
		image = canvas
		output = io.BytesIO()
		image.save(output, format="PNG")
		encoded = base64.b64encode(output.getvalue()).decode("ascii")
		logger.info(
			"thumbnail complete video=%s model=%s quality=%s reference_ms=%.0f openai_ms=%.0f total_ms=%.0f",
			video.id,
			OPENAI_IMAGE_MODEL,
			OPENAI_IMAGE_QUALITY,
			reference_ms,
			openai_ms,
			(time.perf_counter() - started_at) * 1000,
		)
		return f"data:image/png;base64,{encoded}"
	except HTTPException:
		raise
	except Exception as error:
		raise HTTPException(status_code=502, detail="OpenAI thumbnail generation failed.") from error


@app.get("/")
def health_check() -> dict[str, str]:
	return {"name": "CreatorBoost API", "status": "online"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_video(request: AnalyzeRequest, http_request: Request) -> AnalyzeResponse:
	enforce_rate_limit(http_request, "analyze", ANALYZE_RATE_LIMIT)
	youtube_url = str(request.youtube_url)
	video_id = extract_video_id(youtube_url)
	if not video_id:
		raise HTTPException(status_code=422, detail="Provide a valid YouTube video URL.")
	video = get_video_metadata(youtube_url, video_id)
	return AnalyzeResponse(video=video, result=generate_content(video))


@app.post("/generate-thumbnail", response_model=ThumbnailResponse)
async def create_thumbnail(request: ThumbnailRequest, http_request: Request) -> ThumbnailResponse:
	enforce_rate_limit(http_request, "thumbnail", THUMBNAIL_RATE_LIMIT)
	started_at = time.perf_counter()
	youtube_url = str(request.youtube_url)
	video_id = extract_video_id(youtube_url)
	if not video_id:
		raise HTTPException(status_code=422, detail="Provide a valid YouTube video URL.")
	# Always re-fetch metadata server-side instead of trusting the client-supplied
	# `video` field: that field previously let a caller point `thumbnail` at an
	# arbitrary URL (SSRF) and inject an arbitrary title/description/transcript
	# into the image prompt, turning this endpoint into a free-form, unauthenticated
	# OpenAI image-generation proxy.
	video = await asyncio.to_thread(get_video_metadata, youtube_url, video_id)
	logger.info("thumbnail start video=%s concept=%s elapsed_ms=%.0f", video_id, request.concept[:40], (time.perf_counter() - started_at) * 1000)
	image_url = await asyncio.to_thread(generate_thumbnail, video, request.concept)
	return ThumbnailResponse(
		image_url=image_url
	)
