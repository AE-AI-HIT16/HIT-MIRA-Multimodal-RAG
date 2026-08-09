-- Metadata phục vụ truy vấn theo bộ dữ liệu và chuỗi sự kiện lặp lại.
--
-- Quy ước quan trọng:
--   * event_series là loại sự kiện lặp lại (ví dụ "Tuyển thành viên");
--   * event_occurrences là một kỳ cụ thể, khóa nghiệp vụ bằng `label`
--     (ví dụ "HIT-15"), KHÔNG khóa bằng năm;
--   * event_year chỉ là thuộc tính lọc/hiển thị. Một năm có thể có nhiều kỳ,
--     và một kỳ có thể được nhắc trong bài đăng của năm khác;
--   * post không bắt buộc thuộc sự kiện nào. Phần lớn corpus là bài lẻ.

BEGIN;

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            VARCHAR(255) NOT NULL,
    source_type     VARCHAR(50) NOT NULL DEFAULT 'facebook-crawl',
    bucket_name     VARCHAR(100) NOT NULL,
    object_prefix   TEXT NOT NULL,
    source_url      TEXT,
    imported_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT uq_datasets_bucket_prefix UNIQUE (bucket_name, object_prefix)
);

ALTER TABLE posts
    ADD COLUMN IF NOT EXISTS dataset_id UUID;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_posts_dataset'
          AND conrelid = 'posts'::regclass
    ) THEN
        ALTER TABLE posts
            ADD CONSTRAINT fk_posts_dataset
            FOREIGN KEY (dataset_id)
            REFERENCES datasets(dataset_id)
            ON DELETE SET NULL;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_posts_dataset
    ON posts(dataset_id);

CREATE INDEX IF NOT EXISTS idx_posts_created_time
    ON posts(created_time);

CREATE TABLE IF NOT EXISTS event_series (
    series_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slug            VARCHAR(255) NOT NULL UNIQUE,
    canonical_name  VARCHAR(255) NOT NULL,
    description     TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS event_occurrences (
    occurrence_id   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    series_id       UUID NOT NULL,
    label           VARCHAR(255) NOT NULL,
    display_name    VARCHAR(255),
    event_year      SMALLINT,
    starts_at       TIMESTAMP,
    ends_at         TIMESTAMP,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_event_occurrence_series
        FOREIGN KEY (series_id)
        REFERENCES event_series(series_id)
        ON DELETE CASCADE,

    -- Một series có thể có nhiều đợt trong cùng năm. Nhãn thế hệ/kỳ mới là
    -- khóa ổn định, ví dụ HIT-15 hoặc HIT-15-DOT-1.
    CONSTRAINT uq_event_occurrence_series_label UNIQUE (series_id, label),

    CONSTRAINT ck_event_occurrence_year
        CHECK (event_year IS NULL OR event_year BETWEEN 2000 AND 2200),

    CONSTRAINT ck_event_occurrence_dates
        CHECK (starts_at IS NULL OR ends_at IS NULL OR starts_at <= ends_at)
);

CREATE INDEX IF NOT EXISTS idx_event_occurrence_series_year
    ON event_occurrences(series_id, event_year);

CREATE TABLE IF NOT EXISTS event_aliases (
    alias_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    series_id         UUID,
    occurrence_id     UUID,
    alias              VARCHAR(255) NOT NULL,
    normalized_alias   VARCHAR(255) NOT NULL UNIQUE,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_event_alias_series
        FOREIGN KEY (series_id)
        REFERENCES event_series(series_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_event_alias_occurrence
        FOREIGN KEY (occurrence_id)
        REFERENCES event_occurrences(occurrence_id)
        ON DELETE CASCADE,

    -- Alias tổng quát trỏ vào series; alias như "tuyển thành viên 2024"
    -- có thể trỏ thẳng vào occurrence HIT-15. Không cho một alias trỏ cả hai.
    CONSTRAINT ck_event_alias_one_target CHECK (
        (series_id IS NOT NULL AND occurrence_id IS NULL)
        OR (series_id IS NULL AND occurrence_id IS NOT NULL)
    )
);

CREATE TABLE IF NOT EXISTS post_event_occurrences (
    post_id          UUID NOT NULL,
    occurrence_id    UUID NOT NULL,
    confidence       FLOAT,
    assigned_by      VARCHAR(50) NOT NULL DEFAULT 'rule',
    evidence         JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_primary       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (post_id, occurrence_id),

    CONSTRAINT fk_post_event_post
        FOREIGN KEY (post_id)
        REFERENCES posts(post_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_post_event_occurrence
        FOREIGN KEY (occurrence_id)
        REFERENCES event_occurrences(occurrence_id)
        ON DELETE CASCADE,

    CONSTRAINT ck_post_event_confidence
        CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
);

CREATE INDEX IF NOT EXISTS idx_post_event_occurrence
    ON post_event_occurrences(occurrence_id, post_id);

-- Chỉ một occurrence được đánh dấu chính cho mỗi post, nhưng post không có
-- occurrence hoặc có nhiều occurrence phụ đều hợp lệ.
CREATE UNIQUE INDEX IF NOT EXISTS uq_post_event_one_primary
    ON post_event_occurrences(post_id)
    WHERE is_primary;

ALTER TABLE media
    ADD COLUMN IF NOT EXISTS content_sha256 VARCHAR(64);

CREATE UNIQUE INDEX IF NOT EXISTS uq_media_bucket_object
    ON media(bucket_name, object_key);

CREATE INDEX IF NOT EXISTS idx_media_content_sha256
    ON media(content_sha256)
    WHERE content_sha256 IS NOT NULL;

COMMIT;
