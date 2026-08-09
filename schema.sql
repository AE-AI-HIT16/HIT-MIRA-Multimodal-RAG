-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


--------------------------------------------------
-- 1. DATASETS
--------------------------------------------------

CREATE TABLE datasets (
    dataset_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    name VARCHAR(255) NOT NULL,

    source_type VARCHAR(50) NOT NULL DEFAULT 'facebook-crawl',

    bucket_name VARCHAR(100) NOT NULL,

    object_prefix TEXT NOT NULL,

    source_url TEXT,

    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT uq_datasets_bucket_prefix
        UNIQUE(bucket_name, object_prefix)
);


--------------------------------------------------
-- 2. FACEBOOK POSTS
--------------------------------------------------

CREATE TABLE posts (
    post_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    facebook_post_id VARCHAR(255) UNIQUE NOT NULL,

    dataset_id UUID,

    content TEXT,

    author VARCHAR(255),

    post_url TEXT,

    created_time TIMESTAMP,

    crawl_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_posts_dataset
        FOREIGN KEY(dataset_id)
        REFERENCES datasets(dataset_id)
        ON DELETE SET NULL
);


CREATE INDEX idx_posts_dataset
ON posts(dataset_id);


CREATE INDEX idx_posts_created_time
ON posts(created_time);


--------------------------------------------------
-- 3. REPEATING EVENT TAXONOMY
--------------------------------------------------

CREATE TABLE event_series (
    series_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    slug VARCHAR(255) UNIQUE NOT NULL,

    canonical_name VARCHAR(255) NOT NULL,

    description TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE event_occurrences (
    occurrence_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    series_id UUID NOT NULL,

    -- Nhãn thế hệ/kỳ (ví dụ HIT-15), không phải năm đăng bài.
    label VARCHAR(255) NOT NULL,

    display_name VARCHAR(255),

    -- Thuộc tính lọc/hiển thị; không tham gia khóa duy nhất.
    event_year SMALLINT,

    starts_at TIMESTAMP,

    ends_at TIMESTAMP,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_event_occurrence_series
        FOREIGN KEY(series_id)
        REFERENCES event_series(series_id)
        ON DELETE CASCADE,

    CONSTRAINT uq_event_occurrence_series_label
        UNIQUE(series_id, label),

    CONSTRAINT ck_event_occurrence_year
        CHECK(event_year IS NULL OR event_year BETWEEN 2000 AND 2200),

    CONSTRAINT ck_event_occurrence_dates
        CHECK(starts_at IS NULL OR ends_at IS NULL OR starts_at <= ends_at)
);


CREATE INDEX idx_event_occurrence_series_year
ON event_occurrences(series_id, event_year);


CREATE TABLE event_aliases (
    alias_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    series_id UUID,

    occurrence_id UUID,

    alias VARCHAR(255) NOT NULL,

    normalized_alias VARCHAR(255) UNIQUE NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_event_alias_series
        FOREIGN KEY(series_id)
        REFERENCES event_series(series_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_event_alias_occurrence
        FOREIGN KEY(occurrence_id)
        REFERENCES event_occurrences(occurrence_id)
        ON DELETE CASCADE,

    CONSTRAINT ck_event_alias_one_target CHECK(
        (series_id IS NOT NULL AND occurrence_id IS NULL)
        OR (series_id IS NULL AND occurrence_id IS NOT NULL)
    )
);


CREATE TABLE post_event_occurrences (
    post_id UUID NOT NULL,

    occurrence_id UUID NOT NULL,

    confidence FLOAT,

    assigned_by VARCHAR(50) NOT NULL DEFAULT 'rule',

    evidence JSONB NOT NULL DEFAULT '{}'::jsonb,

    is_primary BOOLEAN NOT NULL DEFAULT FALSE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY(post_id, occurrence_id),

    CONSTRAINT fk_post_event_post
        FOREIGN KEY(post_id)
        REFERENCES posts(post_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_post_event_occurrence
        FOREIGN KEY(occurrence_id)
        REFERENCES event_occurrences(occurrence_id)
        ON DELETE CASCADE,

    CONSTRAINT ck_post_event_confidence
        CHECK(confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
);


CREATE INDEX idx_post_event_occurrence
ON post_event_occurrences(occurrence_id, post_id);


CREATE UNIQUE INDEX uq_post_event_one_primary
ON post_event_occurrences(post_id)
WHERE is_primary;



--------------------------------------------------
-- 4. ALL MEDIA OBJECTS
-- image / video / frame
--------------------------------------------------

CREATE TABLE media (

    media_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    post_id UUID NOT NULL,

    media_type VARCHAR(20) NOT NULL,
    -- image
    -- video
    -- frame


    bucket_name VARCHAR(100) DEFAULT 'mira-data',

    object_key TEXT NOT NULL,
    -- MinIO path
    -- videos/post1/video.mp4
    -- images/post1/img1.jpg
    -- frames/video1/frame001.jpg


    content_sha256 VARCHAR(64),


    parent_media_id UUID,
    -- frame -> video
    -- image/video = NULL


    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,


    CONSTRAINT fk_media_post
        FOREIGN KEY(post_id)
        REFERENCES posts(post_id)
        ON DELETE CASCADE,


    CONSTRAINT fk_media_parent
        FOREIGN KEY(parent_media_id)
        REFERENCES media(media_id)
);



CREATE INDEX idx_media_type
ON media(media_type);


CREATE UNIQUE INDEX uq_media_bucket_object
ON media(bucket_name, object_key);


CREATE INDEX idx_media_content_sha256
ON media(content_sha256)
WHERE content_sha256 IS NOT NULL;



--------------------------------------------------
-- 3. VIDEO METADATA
--------------------------------------------------

CREATE TABLE videos (

    video_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    media_id UUID UNIQUE NOT NULL,


    duration FLOAT,

    fps FLOAT,

    raw_frames INTEGER,

    selected_keyframes INTEGER,


    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,


    CONSTRAINT fk_video_media
        FOREIGN KEY(media_id)
        REFERENCES media(media_id)
        ON DELETE CASCADE
);



--------------------------------------------------
-- 4. FRAME METADATA
--------------------------------------------------

CREATE TABLE frames (

    frame_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),


    media_id UUID UNIQUE NOT NULL,
    -- media_type = frame


    video_media_id UUID NOT NULL,
    -- video gốc


    timestamp FLOAT,

    frame_index INTEGER,


    CONSTRAINT fk_frame_media
        FOREIGN KEY(media_id)
        REFERENCES media(media_id)
        ON DELETE CASCADE,


    CONSTRAINT fk_frame_video
        FOREIGN KEY(video_media_id)
        REFERENCES media(media_id)
);



CREATE INDEX idx_frames_video
ON frames(video_media_id);



--------------------------------------------------
-- 5. OCR RESULT
--------------------------------------------------

CREATE TABLE ocr_results (

    ocr_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),


    media_id UUID UNIQUE NOT NULL,


    ocr_status VARCHAR(20) DEFAULT 'PENDING',
    -- PENDING
    -- PROCESSING
    -- DONE
    -- FAILED


    ocr_text TEXT,


    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,


    CONSTRAINT fk_ocr_media
        FOREIGN KEY(media_id)
        REFERENCES media(media_id)
        ON DELETE CASCADE
);



--------------------------------------------------
-- 6. CAPTION RESULT
--------------------------------------------------

CREATE TABLE caption_results (

    caption_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),


    media_id UUID UNIQUE NOT NULL,


    caption_status VARCHAR(20)
        DEFAULT 'PENDING',


    caption_text TEXT,


    caption_model VARCHAR(100),


    vision_metadata JSONB DEFAULT '{}'::jsonb,


    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,


    CONSTRAINT fk_caption_media
        FOREIGN KEY(media_id)
        REFERENCES media(media_id)
        ON DELETE CASCADE
);



--------------------------------------------------
-- 7. OBJECT DETECTION RESULT
--------------------------------------------------

CREATE TABLE object_results (

    object_result_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),


    media_id UUID UNIQUE NOT NULL,


    status VARCHAR(20)
        DEFAULT 'PENDING',


    model VARCHAR(100),


    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,


    CONSTRAINT fk_object_media
        FOREIGN KEY(media_id)
        REFERENCES media(media_id)
        ON DELETE CASCADE
);



--------------------------------------------------
-- 8. DETECTED OBJECTS
--------------------------------------------------

CREATE TABLE detected_objects (

    object_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),


    object_result_id UUID NOT NULL,


    label VARCHAR(100),


    confidence FLOAT,


    x1 FLOAT,
    y1 FLOAT,
    x2 FLOAT,
    y2 FLOAT,


    CONSTRAINT fk_detect_object_result
        FOREIGN KEY(object_result_id)
        REFERENCES object_results(object_result_id)
        ON DELETE CASCADE
);



--------------------------------------------------
-- 9. TRANSCRIPT
--------------------------------------------------

CREATE TABLE transcripts (

    transcript_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),


    video_id UUID UNIQUE NOT NULL,


    status VARCHAR(20)
        DEFAULT 'PENDING',


    language VARCHAR(20),


    model VARCHAR(100),


    full_text TEXT,


    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,


    CONSTRAINT fk_transcript_video
        FOREIGN KEY(video_id)
        REFERENCES videos(video_id)
        ON DELETE CASCADE
);



--------------------------------------------------
-- 10. ASR SEGMENTS
--------------------------------------------------

CREATE TABLE transcript_segments (

    segment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),


    transcript_id UUID NOT NULL,


    start_time FLOAT,

    end_time FLOAT,


    text TEXT,


    CONSTRAINT fk_segment_transcript
        FOREIGN KEY(transcript_id)
        REFERENCES transcripts(transcript_id)
        ON DELETE CASCADE
);



--------------------------------------------------
-- 11. WORKER JOB QUEUE
--------------------------------------------------

CREATE TABLE processing_jobs (

    job_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),


    media_id UUID NOT NULL,


    task_type VARCHAR(50) NOT NULL,
    -- keyframe
    -- ocr
    -- caption
    -- object_detection
    -- asr
    -- embedding


    status VARCHAR(20)
        DEFAULT 'PENDING',


    retry_count INTEGER DEFAULT 0,


    error_message TEXT,


    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,


    started_at TIMESTAMP,


    finished_at TIMESTAMP,


    CONSTRAINT fk_job_media
        FOREIGN KEY(media_id)
        REFERENCES media(media_id)
        ON DELETE CASCADE
);



CREATE INDEX idx_jobs_worker
ON processing_jobs(task_type, status);


--------------------------------------------------
-- 12. VECTOR DATABASE MAPPING
--------------------------------------------------

CREATE TABLE embeddings (

    embedding_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),


    media_id UUID UNIQUE NOT NULL,


    vector_db_id VARCHAR(255),


    model VARCHAR(100),


    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,


    CONSTRAINT fk_embedding_media
        FOREIGN KEY(media_id)
        REFERENCES media(media_id)
        ON DELETE CASCADE
);
