import os

from dotenv import load_dotenv

from hass.client import HassClient
from hass.triggers import triggers as hass_triggers
from magic.caster import Spellcaster
from magic.classifier import SpellClassifier
from magic.detector import SpellDetector

load_dotenv()

# Home Assistant
hass_client = HassClient(os.getenv('HASS_BASE_URL'), os.getenv('HASS_API_TOKEN'))

# Spell Detection

video_src = 0

spell_classifier = SpellClassifier(
    'models/spell_detector_model.h5',
    'models/spell_detector_classes.json'
)

spellcaster = Spellcaster(hass_client, hass_triggers)

SpellDetector(video_src, spell_classifier, spellcaster).run(is_remove_background_enabled=True)