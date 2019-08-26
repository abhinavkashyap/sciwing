import falcon
import parsect.api.conf as config
from parsect.api.resources.sect_label_resource import SectLabelResource
from parsect.api.resources.parscit_tagger_resource import ParscitTaggerResource
from parsect.api.pdf_store import PdfStore
from falcon_multipart.middleware import MultipartMiddleware

# A few initialization
multipart_middleware = MultipartMiddleware()

PDF_STORE_LOCATION = config.PDF_STORE_LOCATION
BIN_FOLDER = config.BIN_FOLDER
PDF_BOX_EXEC = BIN_FOLDER.joinpath("pdfbox-app-2.0.16.jar")

# model paths and infer functions
SECT_LABEL_MODEL_PATH = config.SECT_LABEL_MODEL_PATH
SECT_LABEL_INFER_FUNCTION = config.SECT_LABEL_INFER_FUNCTION

PARSCIT_TAGGER_MODEL_PATH = config.PARSCIT_TAGGER_MODEL_PATH
PARSCIT_TAGGER_INFER_FUNCTION = config.PARSCIT_TAGGER_INFER_FUNCTION

api = application = falcon.API(middleware=[multipart_middleware])
pdf_store = PdfStore(store_path=PDF_STORE_LOCATION)

sect_label_resource = SectLabelResource(
    pdf_store=pdf_store,
    pdfbox_jar_path=PDF_BOX_EXEC,
    model_filepath=str(SECT_LABEL_MODEL_PATH),
    model_infer_func=SECT_LABEL_INFER_FUNCTION,
)

parscit_tagger_resource = ParscitTaggerResource(
    model_filepath=str(PARSCIT_TAGGER_MODEL_PATH),
    model_infer_func=PARSCIT_TAGGER_INFER_FUNCTION,
)

# All routes in the application

# Sect Label specific routes
api.add_route("/sect_label", sect_label_resource)
api.add_route("/parscit_tagger", parscit_tagger_resource)
