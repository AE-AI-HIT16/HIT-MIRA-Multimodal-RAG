-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


--------------------------------------------------
-- 1. FACEBOOK POSTS
--------------------------------------------------

CREATE TABLE posts (
    post_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    facebook_post_id VARCHAR(255) UNIQUE NOT NULL,

    content TEXT,

    author VARCHAR(255),

    post_url TEXT,

    created_time TIMESTAMP,

    crawl_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);



--------------------------------------------------
-- 2. ALL MEDIA OBJECTS
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


    status VARCHAR(20) DEFAULT 'PENDING',
    -- PENDING
    -- PROCESSING
    -- DONE
    -- FAILED


    text TEXT,


    avg_confidence FLOAT,


    model VARCHAR(100),


    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,


    CONSTRAINT fk_ocr_media
        FOREIGN KEY(media_id)
        REFERENCES media(media_id)
        ON DELETE CASCADE
);



--------------------------------------------------
-- 6. OCR BOUNDING BOX
--------------------------------------------------

CREATE TABLE ocr_boxes (

    box_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),


    ocr_id UUID NOT NULL,


    text TEXT,

    confidence FLOAT,


    x1 FLOAT,
    y1 FLOAT,
    x2 FLOAT,
    y2 FLOAT,


    CONSTRAINT fk_box_ocr
        FOREIGN KEY(ocr_id)
        REFERENCES ocr_results(ocr_id)
        ON DELETE CASCADE
);



--------------------------------------------------
-- 7. CAPTION RESULT
--------------------------------------------------

CREATE TABLE caption_results (

    caption_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),


    media_id UUID UNIQUE NOT NULL,


    status VARCHAR(20)
        DEFAULT 'PENDING',


    caption_text TEXT,


    model VARCHAR(100),


    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,


    CONSTRAINT fk_caption_media
        FOREIGN KEY(media_id)
        REFERENCES media(media_id)
        ON DELETE CASCADE
);



--------------------------------------------------
-- 8. OBJECT DETECTION RESULT
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
-- 9. DETECTED OBJECTS
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
-- 10. TRANSCRIPT
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
-- 11. ASR SEGMENTS
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
-- 12. WORKER JOB QUEUE
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
-- 13. VECTOR DATABASE MAPPING
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