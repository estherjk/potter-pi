class Spellcaster:
    """
    The Spellcaster triggers the Home Assistant.
    """

    def __init__(self, hass_client, hass_triggers):
        self.hass_client = hass_client
        self.hass_triggers = hass_triggers

    def cast_spell(self, spell):
        """
        Cast the spell!
        """

        if spell not in self.hass_triggers.keys() or spell == 'unknown':
            print('Spell does not have an associated Hass trigger.')
            return

        self.hass_client.trigger_automation(self.hass_triggers[spell])