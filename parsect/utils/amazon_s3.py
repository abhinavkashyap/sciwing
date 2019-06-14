import boto3
import parsect.constants as constants
import wasabi
import json
from collections import namedtuple
from botocore.exceptions import ClientError
import pathlib
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
        """

        :param filename: str
        The local path filename
        :param obj_name: type: str
        The object name that will be stored in s3
        :return: None
        """
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

    def download_folder(self, folder_name_s3: str):
        bucket = self.s3_resource.Bucket(self.credentials.bucket_name)
        for key in bucket.objects.filter(Prefix=folder_name_s3):
            if not os.path.exists(f"{OUTPUT_DIR}/{os.path.dirname(key.key)}"):
                os.makedirs(f"{OUTPUT_DIR}/{os.path.dirname(key.key)}")
            bucket.download_file(key.key, f"{OUTPUT_DIR}/{key.key}")


if __name__ == "__main__":
    import os

    OUTPUT_DIR = PATHS["OUTPUT_DIR"]
    bow_random_experiment_folder = os.path.join(
        OUTPUT_DIR, "bow_random_emb_lc_4kw_15ml_75d_50e_1e-4lr"
    )

    util = S3Util(
        aws_cred_config_json_filename=os.path.join(
            AWS_CRED_DIR, "aws_s3_credentials.json"
        )
    )

    msg_printer = wasabi.Printer()
    with msg_printer.loading(f"Uploading folder {bow_random_experiment_folder}"):
        util.upload_folder(
            folder_name=bow_random_experiment_folder,
            base_folder_name=os.path.basename(bow_random_experiment_folder),
        )
    msg_printer.good(f"Finished uploading folder {bow_random_experiment_folder}")
