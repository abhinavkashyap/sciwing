import boto3
import sciwing.constants as constants
import wasabi
import json
from collections import namedtuple
from botocore.exceptions import ClientError
import pathlib
import re
import os

PATHS = constants.PATHS
AWS_CRED_DIR = PATHS["AWS_CRED_DIR"]
OUTPUT_DIR = PATHS["OUTPUT_DIR"]


class S3Util:
    def __init__(self, aws_cred_config_json_filename: str):
        self.aws_cred_config_json_filename = aws_cred_config_json_filename
        self.msg_printer = wasabi.Printer()

        self.credentials = self.load_credentials()
        self.s3_client = self.get_client()
        self.s3_resource = self.get_resource()

    def load_credentials(self):
        with open(self.aws_cred_config_json_filename) as fp:
            cred = json.load(fp)

        try:
            aws_access_key_id = cred["aws_access_key_id"]
            aws_access_secret = cred["aws_access_secret"]
            aws_region = cred["region"]
            bucket_name = cred["parsect_bucket_name"]

            Credentials = namedtuple(
                "Credentials", ["access_key", "access_secret", "region", "bucket_name"]
            )
            credentials = Credentials(
                access_key=aws_access_key_id,
                access_secret=aws_access_secret,
                region=aws_region,
                bucket_name=bucket_name,
            )
            return credentials

        except KeyError:
            self.msg_printer.fail(
                f"Your credential file f{self.aws_cred_config_json_filename} "
                f"is malformed. Please contact the developers for more info "
            )

    def get_client(self):
        try:
            s3_client = boto3.client(
                "s3",
                region_name=self.credentials.region,
                aws_access_key_id=self.credentials.access_key,
                aws_secret_access_key=self.credentials.access_secret,
            )
            return s3_client
        except ClientError:
            self.msg_printer.fail(f"Failed to connect to s3 instance")

    def get_resource(self):
        try:
            s3_resource = boto3.resource(
                "s3",
                region_name=self.credentials.region,
                aws_access_key_id=self.credentials.access_key,
                aws_secret_access_key=self.credentials.access_secret,
            )
            return s3_resource
        except ClientError:
            self.msg_printer.fail(f"Failed to get the s3 resource")

    def upload_file(self, filename: str, obj_name: str = None):
        if obj_name is None:
            obj_name = filename

        try:
            self.s3_client.upload_file(filename, self.credentials.bucket_name, obj_name)
        except ClientError:
            self.msg_printer.fail(f"Could not upload file {filename}")

    def upload_folder(self, folder_name: str, base_folder_name: str):
        path = pathlib.Path(folder_name)
        for file in path.iterdir():
            if file.is_file():
                self.upload_file(
                    filename=str(file), obj_name=f"{base_folder_name}/{file.name}"
                )
            elif file.is_dir():
                self.upload_folder(
                    folder_name=str(file),
                    base_folder_name=f"{base_folder_name}/{file.name}",
                )

    def download_file(self, filename_s3: str, local_filename: str):
        object = self.s3_resource.Object(self.credentials.bucket_name, filename_s3)
        object.download_file(local_filename)

    def download_folder(
        self,
        folder_name_s3: str,
        download_only_best_checkpoint: bool = False,
        chkpoints_foldername: str = "checkpoints",
        best_model_filename="best_model.pt",
    ):
        bucket = self.s3_resource.Bucket(self.credentials.bucket_name)
        if len(list(bucket.objects.filter(Prefix=folder_name_s3))) == 0:
            raise FileNotFoundError(f"Failed to find folder {folder_name_s3}")

        for key in bucket.objects.filter(Prefix=folder_name_s3):
            if not os.path.exists(f"{OUTPUT_DIR}/{os.path.dirname(key.key)}"):
                os.makedirs(f"{OUTPUT_DIR}/{os.path.dirname(key.key)}")
            if download_only_best_checkpoint:
                if re.search(chkpoints_foldername, key.key):
                    if re.search(best_model_filename, key.key):
                        bucket.download_file(key.key, f"{OUTPUT_DIR}/{key.key}")
                else:
                    bucket.download_file(key.key, f"{OUTPUT_DIR}/{key.key}")
            else:
                bucket.download_file(key.key, f"{OUTPUT_DIR}/{key.key}")

    def search_folders_with(self, pattern):
        bucket = self.s3_resource.Bucket(self.credentials.bucket_name)
        foldernames = []
        for obj in bucket.objects.all():
            foldernames.append(obj.key.split("/")[0])

        foldernames = list(set(foldernames))
        filtered_folder_names = []
        for foldername in foldernames:
            obj = re.match(pattern, foldername)
            if obj is not None:
                filtered_folder_names.append(foldername)

        return filtered_folder_names
