import boto3
import sciwing.constants as constants
import wasabi
import json
from collections import namedtuple
from botocore.exceptions import ClientError
import pathlib
import re
import os
from typing import NamedTuple

PATHS = constants.PATHS
AWS_CRED_DIR = PATHS["AWS_CRED_DIR"]
OUTPUT_DIR = PATHS["OUTPUT_DIR"]


class S3Util:
    def __init__(self, aws_cred_config_json_filename: str):
        """ Some utilities that would be useful to upload folders/models to s3

        Parameters
        ----------
        aws_cred_config_json_filename : str
            You need to instantiate this file with a aws configuration json file

            The following will be the keys and values
                aws_access_key_id : str
                    The access key id for the AWS account that you have
                aws_access_secret : str
                    The access secret
                region : str
                    The region in which your bucket is present
                parsect_bucket_name : str
                    The name of the bucket where all the models/experiments will be sotred
        """
        self.aws_cred_config_json_filename = aws_cred_config_json_filename
        self.msg_printer = wasabi.Printer()

        self.credentials = self.load_credentials()
        self.s3_client = self.get_client()
        self.s3_resource = self.get_resource()

    def load_credentials(self) -> NamedTuple:
        """ Read the credentials from the json file

        Returns
        -------
        NamedTuple
            a named tuple with access_key, access_secret, region and bucket_name as the keys
            and the corresponding values filled in

        """
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
        """ Returns boto3 client

        Returns
        -------
        boto3.client
            The client object that manages all the aws operations
            The client is the low level access to the connection with s3

        """
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
        """Returns a high level manager for the aws bucket

        Returns
        -------
        boto3.resource
            Resource that manages connections with s3

        """
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
        Parameters
        ----------
        filename : str
            The filename in the local directory that needs to be  uploaded to s3
        obj_name : str
            The filename to be used in s3 bucket. If None then obj_name in s3 will be
            the same as the filename

        """
        if obj_name is None:
            obj_name = filename

        try:
            self.s3_client.upload_file(filename, self.credentials.bucket_name, obj_name)
        except ClientError:
            self.msg_printer.fail(f"Could not upload file {filename}")

    def upload_folder(self, folder_name: str, base_folder_name: str):
        """ Recursively uploads a folder to s3

        Parameters
        ----------
        folder_name : str
            The name of the local folder that is uploaded
        base_folder_name : str
            The name of the folder from which the current folder
            being uploaded stems from. This is needed to associate appropriate
            files and directories to their hierarchies within the folder

        """
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
        """ Downloads a file from s3

        Parameters
        ----------
        filename_s3 : str
            A filename in s3 that needs to be downloaded
        local_filename : str
            The local filename that will be used

        """
        object = self.s3_resource.Object(self.credentials.bucket_name, filename_s3)
        object.download_file(local_filename)

    def download_folder(
        self,
        folder_name_s3: str,
        download_only_best_checkpoint: bool = False,
        chkpoints_foldername: str = "checkpoints",
        best_model_filename="best_model.pt",
        output_dir: str = OUTPUT_DIR,
    ):
        """ Downloads a folder from s3 recursively

        Parameters
        ----------
        folder_name_s3 : str
            The name of the folder in s3
        download_only_best_checkpoint : bool
            If the folder being downloaded is an experiment folder, then you
            can download only the best model checkpoints for running test or inference
        chkpoints_foldername : str
            The name of the checkpoints folder where the best model parameters are stored
        best_model_filename : str
            The name of the file where the best model parameters are stored

        Returns
        -------

        """
        bucket = self.s3_resource.Bucket(self.credentials.bucket_name)
        with self.msg_printer.loading(f"Downloading folder {folder_name_s3}"):
            if len(list(bucket.objects.filter(Prefix=folder_name_s3))) == 0:
                raise FileNotFoundError(f"Failed to find folder {folder_name_s3}")

            for key in bucket.objects.filter(Prefix=folder_name_s3):
                if not os.path.exists(f"{output_dir}/{os.path.dirname(key.key)}"):
                    os.makedirs(f"{output_dir}/{os.path.dirname(key.key)}")
                if download_only_best_checkpoint:
                    if re.search(chkpoints_foldername, key.key):
                        if re.search(best_model_filename, key.key):
                            bucket.download_file(key.key, f"{output_dir}/{key.key}")
                    else:
                        bucket.download_file(key.key, f"{output_dir}/{key.key}")
                else:
                    bucket.download_file(key.key, f"{output_dir}/{key.key}")
        self.msg_printer.good(f"Finished downloading {folder_name_s3}")

    def search_folders_with(self, pattern):
        """ Searches for folders in the s3 bucket with specific pattern

        Parameters
        ----------
        pattern : str
            A regex pattern

        Returns
        -------
        List[str]
            The list of foldernames that match the pattern

        """
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
