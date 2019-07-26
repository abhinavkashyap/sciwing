import falcon
import parsect.api.conf as config
from parsect.api.resources.sect_label_resource import SectLabelResource
from parsect.api.pdf_store import PdfStore
from falcon_multipart.middleware import MultipartMiddleware

# A few initialization
PDF_STORE_LOCATION = config.PDF_STORE_LOCATION
BIN_FOLDER = config.BIN_FOLDER
PDF_BOX_EXEC = BIN_FOLDER.joinpath("pdfbox-app-2.0.16.jar")
multipart_middleware = MultipartMiddleware()
SECT_MODEL_PATH = config.SECT_MODEL_PATH
SECT_LABEL_INFER_FUNCTION = config.SECT_LABEL_INFER_FUNCTION

api = application = falcon.API(middleware=[multipart_middleware])
pdf_store = PdfStore(store_path=PDF_STORE_LOCATION)

sect_label_resource = SectLabelResource(
    pdf_store=pdf_store, pdfbox_jar_path=PDF_BOX_EXEC
)

# All routes in the application

# Sect Label specific routes
api.add_route("/sect_label", sect_label_resource)
