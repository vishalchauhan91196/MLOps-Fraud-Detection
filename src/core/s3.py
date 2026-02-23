import json
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from src.core.exceptions import PipelineIOError


def get_s3_client(region_name: str):
    """Return s3 client derived from the provided inputs and runtime context.

    Encapsulates lookup and fallback logic in a single reusable call.
    """
    try:
        return boto3.client("s3", region_name=region_name)
    except Exception as exc:
        raise PipelineIOError("Failed to initialize S3 client") from exc


def get_json_if_exists(client, bucket: str, key: str) -> Optional[dict]:
    """Return json if exists derived from the provided inputs and runtime context.

    Encapsulates lookup and fallback logic in a single reusable call.
    """
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            return None
        raise PipelineIOError(f"Failed to read S3 object s3://{bucket}/{key}") from exc
    except Exception as exc:
        raise PipelineIOError(f"Failed to parse S3 JSON object s3://{bucket}/{key}") from exc


def upload_file(client, local_path: str, bucket: str, key: str) -> None:
    """Persist file to the target destination.

    Creates or overwrites the target artifact needed by later pipeline stages.
    """
    try:
        client.upload_file(local_path, bucket, key)
    except Exception as exc:
        raise PipelineIOError(f"Failed to upload file to s3://{bucket}/{key}") from exc


def put_json(client, bucket: str, key: str, payload: dict) -> None:
    """Persist json to the target destination.

    Creates or overwrites the target artifact needed by later pipeline stages.
    """
    try:
        client.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload, indent=2).encode("utf-8"))
    except Exception as exc:
        raise PipelineIOError(f"Failed to write JSON to s3://{bucket}/{key}") from exc

