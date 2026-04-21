import "server-only";
import {
  S3Client,
  GetObjectCommand,
  PutObjectCommand,
  DeleteObjectCommand,
  DeleteObjectsCommand,
  ListObjectsV2Command,
  HeadBucketCommand,
  NoSuchKey,
} from "@aws-sdk/client-s3";

/**
 * Cloudflare R2 client wrapper.
 *
 * R2 is S3-compatible. We use the AWS SDK v3 S3 client against the
 * R2 endpoint. All credentials must come from env vars — never hard-code.
 */

function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v || v.trim() === "") {
    throw new Error(
      `Missing required env var ${name}. Set it in .env.local (see .env.example).`,
    );
  }
  return v;
}

// Lazy singleton so the client is only constructed when R2 is actually used.
let _client: S3Client | null = null;
function client(): S3Client {
  if (_client) return _client;
  _client = new S3Client({
    region: "auto", // R2 uses "auto"
    endpoint: requireEnv("R2_ENDPOINT"),
    credentials: {
      accessKeyId: requireEnv("R2_ACCESS_KEY_ID"),
      secretAccessKey: requireEnv("R2_SECRET_ACCESS_KEY"),
    },
  });
  return _client;
}

export const BUCKET = () => requireEnv("R2_BUCKET");
export const PREFIX_RAW = () => process.env.R2_PREFIX_RAW || "raw";
export const PREFIX_ANN = () => process.env.R2_PREFIX_ANNOTATIONS || "annotations";
export const PREFIX_EXP = () => process.env.R2_PREFIX_EXPORTS || "exports";

/** Join prefix + key safely (no double slashes, no leading slash). */
export function joinKey(...parts: string[]): string {
  return parts
    .map((p) => p.replace(/^\/+|\/+$/g, ""))
    .filter(Boolean)
    .join("/");
}

/** Verify credentials and bucket access. Useful as a startup/health check. */
export async function pingBucket(): Promise<void> {
  await client().send(new HeadBucketCommand({ Bucket: BUCKET() }));
}

/** Put an object. body can be Buffer, Uint8Array, or string. */
export async function putObject(
  key: string,
  body: Buffer | Uint8Array | string,
  contentType?: string,
): Promise<void> {
  await client().send(
    new PutObjectCommand({
      Bucket: BUCKET(),
      Key: key,
      Body: body,
      ContentType: contentType,
    }),
  );
}

/** Get an object body as a Buffer. Returns null if the key does not exist. */
export async function getObjectBuffer(key: string): Promise<{ body: Buffer; contentType?: string } | null> {
  try {
    const out = await client().send(
      new GetObjectCommand({ Bucket: BUCKET(), Key: key }),
    );
    if (!out.Body) return null;
    // AWS SDK v3 Body is a ReadableStream / Blob / Readable. transformToByteArray() handles all.
    const bytes = await (out.Body as unknown as { transformToByteArray: () => Promise<Uint8Array> })
      .transformToByteArray();
    return { body: Buffer.from(bytes), contentType: out.ContentType };
  } catch (err: unknown) {
    if (err instanceof NoSuchKey) return null;
    // Some S3-compatible stores return a 404 without NoSuchKey subclass
    const httpStatus = (err as { $metadata?: { httpStatusCode?: number } })?.$metadata?.httpStatusCode;
    if (httpStatus === 404) return null;
    throw err;
  }
}

/** Get an object body as a UTF-8 string. Returns null if missing. */
export async function getObjectText(key: string): Promise<string | null> {
  const res = await getObjectBuffer(key);
  return res ? res.body.toString("utf8") : null;
}

/** Delete a single object. Idempotent — does not throw if key is absent. */
export async function deleteObject(key: string): Promise<void> {
  await client().send(new DeleteObjectCommand({ Bucket: BUCKET(), Key: key }));
}

/** Delete up to thousands of objects in batched requests (1 000 per call). */
export async function deleteObjects(keys: string[]): Promise<number> {
  if (keys.length === 0) return 0;
  let deleted = 0;
  for (let i = 0; i < keys.length; i += 1000) {
    const chunk = keys.slice(i, i + 1000);
    await client().send(
      new DeleteObjectsCommand({
        Bucket: BUCKET(),
        Delete: { Objects: chunk.map((Key) => ({ Key })), Quiet: true },
      }),
    );
    deleted += chunk.length;
  }
  return deleted;
}

/** List all object keys under a prefix. Handles pagination. */
export async function listKeys(prefix: string): Promise<string[]> {
  const keys: string[] = [];
  let token: string | undefined = undefined;
  // Make sure prefix ends with / so we don't match sibling prefixes
  const listPrefix = prefix.endsWith("/") ? prefix : prefix + "/";
  do {
    const out: {
      Contents?: { Key?: string }[];
      NextContinuationToken?: string;
      IsTruncated?: boolean;
    } = await client().send(
      new ListObjectsV2Command({
        Bucket: BUCKET(),
        Prefix: listPrefix,
        ContinuationToken: token,
      }),
    );
    for (const obj of out.Contents || []) {
      if (obj.Key) keys.push(obj.Key);
    }
    token = out.IsTruncated ? out.NextContinuationToken : undefined;
  } while (token);
  return keys;
}

/**
 * Ensure the folder "markers" exist in the bucket. S3/R2 has no real folders,
 * but writing zero-byte objects with trailing slashes makes the structure
 * visible in dashboards and protects against accidental typos later.
 */
export async function ensurePrefixes(): Promise<void> {
  const markers = [PREFIX_RAW(), PREFIX_ANN(), PREFIX_EXP()].map((p) => `${p}/`);
  await Promise.all(markers.map((k) => putObject(k, "", "application/x-directory")));
}
