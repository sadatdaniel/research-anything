"""How chroniclers write: phrasing banks for every kind of event, in three registers.

The same event kind is written many different ways, so plain keyword matching can't line up
occurrences. Common words ("star", "tail", "flood", "hunger", "fire") also appear in unrelated
records, as they would in real chronicles.

Slots in {braces} are filled by the generator. Place-like slots ({city}, {river}, {region}, {coast})
use whichever of the place's two names the chronicler prefers.
"""

from __future__ import annotations

REGISTERS = ("plain", "florid", "terse")

# Each variant uses different key words, so no single word marks a direction or a colour.
DIRECTION_PHRASES = {
    "west": ["in the west", "towards the setting sun", "over the evening hills", "where the day goes down",
             "above the sunset"],
    "east": ["in the east", "above the morning sea", "towards the sunrise", "over the dawn ridge",
             "where the day is born"],
    "north": ["in the north", "beneath the Wain", "over the frozen marshes", "towards the cold quarter",
              "above the pole"],
    "south": ["in the south", "above the noon road", "over the warm hills", "towards the desert quarter",
              "low over the midday sea"],
}

COLOR_PHRASES = {
    "pale": ["pale", "milky", "ash-coloured", "bone-coloured"],
    "red": ["red", "blood-coloured", "ruddy", "ember-bright"],
    "golden": ["golden", "yellow", "honey-coloured", "brassy"],
    "green": ["green", "sickly", "leek-coloured"],
    "white": ["white", "snowy", "chalky", "silver"],
    "brown": ["brown", "dun", "earthen"],
}

DATE_FORMATS = {
    "plain": ["In the year {year}, {s}", "In {year}, {s}", "That year, {year}, {s}"],
    "florid": ["In the year of grace {year}, {s}", "In the year {year} of the Reckoning, {s}",
               "Be it remembered that in the year {year} {s}"],
    "terse": ["{year}. {S}", "Year {year}. {S}"],
}

# kind -> register -> templates. Each kind lists the slots it needs.
KINDS: dict[str, dict] = {
    # ---- sky ------------------------------------------------------------------------------------
    "comet": {
        "slots": ["dir", "color", "nights"],
        "plain": ["a {color} star with a long tail was seen {dir} for {nights} nights",
                  "a {color} comet appeared {dir} and stayed {nights} nights",
                  "a {color} light like a torch hung {dir} for {nights} nights"],
        "florid": ["a hairy star, {color}, burned {dir} for {nights} nights, and many feared it",
                   "heaven showed a bearded light, {color}, {dir}, which shone {nights} nights",
                   "a {color} fiery sword hung in the sky {dir} for {nights} nights",
                   "a {color} broom of fire swept the heavens {dir} for {nights} nights"],
        "terse": ["Tailed star {dir}, {color}, {nights} nights.", "Comet {dir}, {color}. {nights} nights.",
                  "Torch in the sky {dir}, {color}, {nights} nights.", "Bearded light {dir}. {color}. {nights} nights."],
    },
    "falling_stars": {
        "slots": ["dir"],
        "plain": ["many stars fell from the sky in one night", "stars were seen falling {dir} like rain"],
        "florid": ["the stars of heaven came down {dir} like sparks from a forge",
                   "it seemed the sky itself was weeping stars {dir}"],
        "terse": ["Stars fell.", "Shower of falling stars {dir}."],
    },
    "new_star": {
        "slots": ["dir", "weeks"],
        "plain": ["a new star was seen {dir} that had not been there before, and it stayed {weeks} weeks",
                  "a strange still star stood {dir} for {weeks} weeks"],
        "florid": ["a stranger among the stars took its seat {dir} and kept it {weeks} weeks"],
        "terse": ["New star {dir}, {weeks} weeks."],
    },
    "eclipse": {
        "slots": [],
        "plain": ["the sun was darkened at midday", "the sun grew dark and stars showed at noon"],
        "florid": ["the sun hid its face at noon and the birds fell silent"],
        "terse": ["Sun darkened at noon.", "Eclipse."],
    },
    "red_moon": {
        "slots": ["city"],
        "plain": ["the moon rose red over {city}", "the moon was red as blood above {city}"],
        "florid": ["the moon put on a robe of blood above {city}"],
        "terse": ["Red moon over {city}."],
    },
    # ---- land and water -------------------------------------------------------------------------
    "flood": {
        "slots": ["river", "region"],
        "plain": ["the {river} burst its banks and drowned the fields of {region}",
                  "the waters of the {river} rose higher than any could remember and covered {region}",
                  "there was a great flood on the {river}, and the lower fields of {region} were lost"],
        "florid": ["the {river} forgot its bed and wandered over {region}, drowning the sheaves",
                   "the {river} swelled like a wrathful beast and swallowed the meadows of {region}"],
        "terse": ["Flood on the {river}. {region} under water.", "{river} over its banks."],
    },
    "drought": {
        "slots": ["region", "city"],
        "plain": ["no rain fell in {region} from spring to autumn", "the wells of {city} ran dry"],
        "florid": ["the heavens shut their doors over {region} and not a drop fell all summer"],
        "terse": ["Drought in {region}.", "No rain. Wells of {city} dry."],
    },
    "hard_winter": {
        "slots": ["river"],
        "plain": ["the winter was hard and the {river} froze from bank to bank",
                  "snow lay on the banks of the {river} until the month of flowers"],
        "florid": ["winter laid an iron hand on the land and the {river} became a road of ice"],
        "terse": ["Great frost. {river} frozen.", "Hard winter on the {river}."],
    },
    "storm": {
        "slots": ["coast"],
        "plain": ["a great storm came out of the sea and broke the boats at {coast}",
                  "a storm from the sea drove the waves over the shore at {coast}"],
        "florid": ["the sea rose up in anger against {coast} and the boats were split like kindling"],
        "terse": ["Storm at {coast}. Boats lost.", "Sea storm, {coast}."],
    },
    "earthquake": {
        "slots": ["city", "building"],
        "plain": ["the earth shook at {city} and the tower of the {building} fell"],
        "florid": ["the ground under {city} trembled like a frightened horse, and the {building} cracked"],
        "terse": ["Earthquake at {city}."],
    },
    "locusts": {
        "slots": ["region", "color"],
        "plain": ["{color} locusts came over {region} and ate everything green",
                  "a cloud of {color} grasshoppers settled on {region}",
                  "{color} hoppers rose out of the ground in {region} and stripped the fields"],
        "florid": ["a {color} host of locusts marched over {region} and left the fields bare as a threshing floor",
                   "a {color} swarm of devouring insects darkened the sky over {region}",
                   "the winged plague, {color}, came down on {region}"],
        "terse": ["Locusts, {color}, in {region}.", "{color} swarm. {region} stripped.", "Grasshoppers, {color}, {region}."],
    },
    "dike_break": {
        "slots": ["city"],
        "plain": ["the sea broke the dikes at {city} and salt water covered the fields",
                  "the dikes of {city} gave way and the sea came in"],
        "florid": ["the sea tore open the walls of earth at {city} and poured its salt upon the fields"],
        "terse": ["Dikes broken at {city}.", "Sea in at {city}."],
    },
    "harvest_fail": {
        "slots": ["city", "region"],
        "plain": ["salt spoiled the fields of {city} and nothing grew", "the harvest failed in {region}"],
        "florid": ["the fields of {city} gave back no bread, for the salt had poisoned them"],
        "terse": ["No harvest at {city}.", "Harvest failed, {region}."],
    },
    # ---- people ---------------------------------------------------------------------------------
    "famine": {
        "slots": ["city", "region"],
        "plain": ["there was hunger in {city}, and the poor ate bark",
                  "the granaries of {city} stood empty", "there was a great dearth of bread in {region}"],
        "florid": ["want sat at every table in {city}, and mothers boiled grass for their children",
                   "the bread of {city} ran out, and men fought over a sack of bran"],
        "terse": ["Famine in {city}. Grain dear.", "Hunger at {city}."],
    },
    "plague": {
        "slots": ["city", "region"],
        "plain": ["a pestilence came to {city} and many died", "the coughing sickness spread through {region}"],
        "florid": ["death walked the streets of {city} with a full sack"],
        "terse": ["Pestilence at {city}.", "Sickness in {region}."],
    },
    "fire": {
        "slots": ["city", "building"],
        "plain": ["fire burned the lower town of {city}", "a great fire broke out in {city} and the {building} was lost"],
        "florid": ["flames ate the roofs of {city} from one gate to the other"],
        "terse": ["Fire at {city}.", "{city} burned. The {building} lost."],
    },
    "revolt": {
        "slots": ["city", "ruler"],
        "plain": ["the people of {city} rose against {ruler}", "there was a rising in {city} over the new tax"],
        "florid": ["{city} shut its gates against {ruler} and the bells rang for rebellion"],
        "terse": ["Revolt at {city}."],
    },
    "battle": {
        "slots": ["ruler", "enemy", "place", "outcome"],
        "plain": ["{ruler} fought the men of {enemy} at {place} and {outcome}"],
        "florid": ["{ruler} met the host of {enemy} by {place}, and {outcome}"],
        "terse": ["Battle at {place} with {enemy}; {ruler} {outcome}."],
    },
    "death_ruler": {
        "slots": ["ruler", "city"],
        "plain": ["{ruler} died and was buried at {city}"],
        "florid": ["{ruler} went the way of all flesh and was laid to rest in {city}"],
        "terse": ["Death of {ruler}."],
    },
    "fair": {
        "slots": ["city", "place"],
        "plain": ["a great fair was held at {city}, and merchants came from {place}",
                  "a flood of pilgrims came to {city} for the feast"],
        "florid": ["merchants from {place} filled the squares of {city} with silk and pepper"],
        "terse": ["Fair at {city}.", "Pilgrims at {city}."],
    },
    "building": {
        "slots": ["river", "city", "building"],
        "plain": ["the new bridge over the {river} was finished", "work began on the {building} at {city}"],
        "florid": ["the masons of {city} raised the {building} to the glory of heaven"],
        "terse": ["Bridge over the {river} finished.", "{building} begun at {city}."],
    },
    "embassy": {
        "slots": ["place", "ruler"],
        "plain": ["envoys came from {place} bearing gifts for {ruler}"],
        "florid": ["the lords of {place} sent envoys to {ruler} with gifts of ivory and amber"],
        "terse": ["Envoys from {place}."],
    },
    "migration": {
        "slots": ["city"],
        "plain": ["many families left {city} for the hills", "the people of {city} went away to seek new land"],
        "florid": ["{city} emptied like a broken jar, and its folk scattered to the hills"],
        "terse": ["{city} emptied.", "Folk of {city} left for the hills."],
    },
    "wedding": {
        "slots": ["ruler", "place"],
        "plain": ["{ruler} took a wife from {place}"],
        "florid": ["{ruler} was wed to a daughter of {place}, and there was feasting for a week"],
        "terse": ["{ruler} married."],
    },
}

# Actions a named person can be the subject of (hidden-identity pattern). {who} is the alias used.
PERSON_ACTIONS = {
    "plain": ["{who} led an army against {enemy} near {place}", "{who} gave land to the abbey at {city}",
              "{who} built a hall at {city}", "{who} made peace with {enemy}",
              "{who} was wounded in a quarrel at {city}", "{who} hanged the bandits of {place}"],
    "florid": ["{who} rode against {enemy} with a hundred lances by {place}",
               "{who} endowed the brothers at {city} with three villages",
               "{who} raised a great hall at {city} for feasting"],
    "terse": ["{who} against {enemy} at {place}.", "{who} gave land at {city}.", "{who} at {city}."],
}

# Sentences that belong to no pattern, some reusing pattern words on purpose.
FLAVOR = [
    "The abbot fell sick in the winter.", "A two-headed calf was born at {place}.", "The price of salt rose.",
    "The star of the house of {family} was rising at court.", "The tail of the procession stretched to the river.",
    "There was hunger for war among the young lords.", "The bells were recast.", "Nothing else of note.",
    "A flood of complaints reached the court.", "The old well was cleaned.", "Wolves were seen near {place}.",
    "The tower clock stopped and was mended.", "Many said it was an omen.", "The chronicler's hand grew weary.",
    "Grain was cheap that year.", "Some say this happened a year later.",
]

OUTCOMES = ["won the field", "was driven back", "made a truce", "lost many men"]
BUILDINGS = ["cathedral", "great hall", "mint", "granary", "watchtower", "abbey church", "guildhall"]
