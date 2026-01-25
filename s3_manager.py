#!/usr/bin/env python3
"""
S3 Manager for video upload and storage
Handles AWS S3 bucket operations for the PTZ camera tracking system
"""

import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from config import cfg


class S3Manager:
    """Manages S3 bucket operations for video storage"""

    def __init__(self):
        """Initialize S3 connection with credentials from config"""
        self.region_name = cfg.s3.region
        self.bucket_name = cfg.s3.bucket_name

        try:
            # Initialize S3 client
            self.s3_client = boto3.client(
                's3',
                region_name=self.region_name,
                aws_access_key_id=cfg.s3.access_key_id,
                aws_secret_access_key=cfg.s3.secret_access_key
            )

            # Verify bucket access
            self._verify_bucket_access()

        except NoCredentialsError:
            raise Exception("AWS credentials not found or invalid")
        except Exception as e:
            raise Exception(f"S3 connection error: {e}")

    def _verify_bucket_access(self):
        """Verify that we can access the configured bucket"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise Exception(f"Bucket '{self.bucket_name}' not found")
            elif error_code == '403':
                raise Exception(f"Access denied to bucket '{self.bucket_name}'")
            else:
                raise Exception(f"Error accessing bucket: {e}")

    def upload_file(self, destination_path, source_path, extra_args=None):
        """
        Upload a file to S3 bucket

        Args:
            destination_path: S3 key (path) for the uploaded file
            source_path: Local file path to upload
            extra_args: Additional S3 upload arguments (e.g., ContentType)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure file exists
            if not os.path.exists(source_path):
                print(f"Source file not found: {source_path}")
                return False

            # Set default content type if not provided
            if extra_args is None:
                extra_args = {}

            # Upload file
            self.s3_client.upload_file(
                source_path,
                self.bucket_name,
                destination_path,
                ExtraArgs=extra_args
            )

            return True

        except ClientError as e:
            print(f"Failed to upload file: {e}")
            return False
        except Exception as e:
            print(f"Upload error: {e}")
            return False

    def download_file(self, s3_path, local_path):
        """
        Download a file from S3 bucket

        Args:
            s3_path: S3 key (path) of the file to download
            local_path: Local path where file will be saved

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Download file
            self.s3_client.download_file(
                self.bucket_name,
                s3_path,
                local_path
            )

            return True

        except ClientError as e:
            print(f"Failed to download file: {e}")
            return False
        except Exception as e:
            print(f"Download error: {e}")
            return False

    def delete_object(self, object_key):
        """
        Delete an object from S3 bucket

        Args:
            object_key: S3 key of the object to delete

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            return True

        except ClientError as e:
            print(f"Failed to delete object: {e}")
            return False

    def list_objects(self, prefix="", max_results=100):
        """
        List objects in S3 bucket with optional prefix filter

        Args:
            prefix: Filter objects by prefix (folder path)
            max_results: Maximum number of results to return

        Returns:
            list: List of object keys
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_results
            )

            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            else:
                return []

        except ClientError as e:
            print(f"Failed to list objects: {e}")
            return []

    def get_public_url(self, object_key):
        """
        Get public URL for an S3 object

        Args:
            object_key: S3 key of the object

        Returns:
            str: Public URL
        """
        return f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{object_key}"

    def get_presigned_url(self, object_key, expiration=None):
        """
        Generate a presigned URL for temporary access

        Args:
            object_key: S3 key of the object
            expiration: URL expiration time in seconds

        Returns:
            str: Presigned URL or None if error
        """
        if expiration is None:
            expiration = cfg.s3.ExpiresIn

        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': object_key
                },
                ExpiresIn=expiration
            )
            return url

        except ClientError as e:
            print(f"Failed to generate presigned URL: {e}")
            return None

    def object_exists(self, object_key):
        """
        Check if an object exists in S3 bucket

        Args:
            object_key: S3 key of the object

        Returns:
            bool: True if exists, False otherwise
        """
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            return True

        except ClientError:
            return False


def test_s3_connection():
    """Test S3 connection and basic operations"""
    print("Testing S3 connection...")

    try:
        # Initialize S3 manager
        s3_manager = S3Manager()
        print(f"✅ Connected to S3 bucket: {s3_manager.bucket_name}")

        # List some objects
        print("\nListing objects in 'recordings/' folder:")
        objects = s3_manager.list_objects(prefix="recordings/", max_results=5)

        if objects:
            for obj in objects:
                print(f"  - {obj}")
        else:
            print("  No objects found")

        print("\n✅ S3 connection test successful!")

    except Exception as e:
        print(f"❌ S3 connection test failed: {e}")


if __name__ == '__main__':
    test_s3_connection()