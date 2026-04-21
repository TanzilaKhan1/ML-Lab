// Delete all objects under raw/ and annotations/ in R2.
// Usage:
//   cd annotator
//   node --env-file=.env.local scripts/clear-r2.mjs

import {
  S3Client,
  ListObjectsV2Command,
  DeleteObjectsCommand,
} from "@aws-sdk/client-s3";

const BUCKET = process.env.R2_BUCKET;
const PREFIX_RAW = process.env.R2_PREFIX_RAW || "raw";
const PREFIX_ANN = process.env.R2_PREFIX_ANNOTATIONS || "annotations";

const s3 = new S3Client({
  region: "auto",
  endpoint: process.env.R2_ENDPOINT,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
  },
});

async function listAll(prefix) {
  const keys = [];
  let token;
  do {
    const res = await s3.send(new ListObjectsV2Command({
      Bucket: BUCKET,
      Prefix: prefix + "/",
      ContinuationToken: token,
    }));
    for (const obj of res.Contents ?? []) {
      if (obj.Key) keys.push(obj.Key);
    }
    token = res.IsTruncated ? res.NextContinuationToken : undefined;
  } while (token);
  return keys;
}

async function deleteAll(keys) {
  if (keys.length === 0) return 0;
  let deleted = 0;
  for (let i = 0; i < keys.length; i += 1000) {
    const chunk = keys.slice(i, i + 1000);
    await s3.send(new DeleteObjectsCommand({
      Bucket: BUCKET,
      Delete: { Objects: chunk.map((Key) => ({ Key })), Quiet: true },
    }));
    deleted += chunk.length;
    process.stdout.write(`  deleted ${deleted}/${keys.length}\r`);
  }
  return deleted;
}

async function main() {
  console.log(`Listing raw/ …`);
  const rawKeys = await listAll(PREFIX_RAW);
  console.log(`  ${rawKeys.length} objects`);

  console.log(`Listing annotations/ …`);
  const annKeys = await listAll(PREFIX_ANN);
  console.log(`  ${annKeys.length} objects`);

  const all = [...rawKeys, ...annKeys];
  if (all.length === 0) {
    console.log("Nothing to delete.");
    return;
  }

  console.log(`\nDeleting ${all.length} objects…`);
  const n = await deleteAll(all);
  console.log(`\n✓ Deleted ${n} objects.`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
