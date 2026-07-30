BEGIN;

DROP TABLE IF EXISTS ocr_boxes;

ALTER TABLE ocr_results
    DROP COLUMN IF EXISTS avg_confidence,
    DROP COLUMN IF EXISTS model;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ocr_results' AND column_name = 'status'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ocr_results' AND column_name = 'ocr_status'
    ) THEN
        ALTER TABLE ocr_results RENAME COLUMN status TO ocr_status;
    ELSIF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ocr_results' AND column_name = 'status'
    ) THEN
        UPDATE ocr_results
        SET ocr_status = COALESCE(ocr_status, status);

        ALTER TABLE ocr_results DROP COLUMN status;
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ocr_results' AND column_name = 'text'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ocr_results' AND column_name = 'ocr_text'
    ) THEN
        ALTER TABLE ocr_results RENAME COLUMN text TO ocr_text;
    ELSIF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ocr_results' AND column_name = 'text'
    ) THEN
        UPDATE ocr_results
        SET ocr_text = COALESCE(ocr_text, text);

        ALTER TABLE ocr_results DROP COLUMN text;
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'caption_results' AND column_name = 'status'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'caption_results' AND column_name = 'caption_status'
    ) THEN
        ALTER TABLE caption_results RENAME COLUMN status TO caption_status;
    ELSIF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'caption_results' AND column_name = 'status'
    ) THEN
        UPDATE caption_results
        SET caption_status = COALESCE(caption_status, status);

        ALTER TABLE caption_results DROP COLUMN status;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'caption_results'
          AND column_name = 'model'
    ) AND NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'caption_results'
          AND column_name = 'caption_model'
    ) THEN
        ALTER TABLE caption_results RENAME COLUMN model TO caption_model;
    ELSIF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'caption_results'
          AND column_name = 'model'
    ) THEN
        UPDATE caption_results
        SET caption_model = COALESCE(caption_model, model);

        ALTER TABLE caption_results DROP COLUMN model;
    END IF;
END $$;

ALTER TABLE caption_results
    ADD COLUMN IF NOT EXISTS vision_metadata JSONB DEFAULT '{}'::jsonb;

COMMIT;
