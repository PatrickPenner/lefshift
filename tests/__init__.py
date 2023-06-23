"""Test registration"""
import logging

from .application_utils_tests import ApplicationUtilsTests
from .fp_tests import LEFFingerprintTests
from .model_tests import ModelTests
from .utils_tests import UtilsTests
from .main_tests import MainTests

logging.disable(logging.CRITICAL)  # suppress noise form failure tests
