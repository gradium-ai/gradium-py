# Text Rewriting Rules

The text-to-speech API supports text rewriting rules that normalize and expand certain patterns in the input text before synthesis. These rules help the TTS model properly pronounce dates, times, numbers, email addresses, URLs, phone numbers, and alphanumeric codes.

## Configuration

Rewrite rules can be enabled by adding a `rewrite_rules` field to the `json_config` in the setup message. The field accepts a comma-delimited string of rule names or language aliases.

**Example setup message:**
```json
{
  "json_config": {
    "rewrite_rules": "en"
  }
}
```

Or with specific rules:
```json
{
  "json_config": {
    "rewrite_rules": "TimeEn,Date,NumberEn,EmailEn"
  }
}
```

## Language Aliases

For convenience, language aliases are provided that enable all recommended rules for a specific language:

| Alias | Enabled Rules |
|-------|---------------|
| `en` | TimeEn, Date, AlNum, NumberEn, EmailEn, UrlEn, PhoneEn |
| `fr` | TimeFr, Date, AlNum, NumberFr, EmailFr, UrlFr, PhoneFr |
| `de` | TimeDe, Date, AlNum, NumberDe, EmailDe, UrlDe, PhoneDe |
| `es` | Date, AlNum, NumberEs, EmailEs, UrlEs, PhoneEs |
| `pt` | Date, AlNum, NumberPt, EmailPt, UrlPt, PhonePt |

## Available Rewrite Rules

### Date Rule

**Rule name:** `Date`

Converts numeric dates to a more speech-friendly format.

**Examples:**
- `12/31/2020` → `12-31 2020`
- `16/01/1980` → `16-01 1980`
- `1/5.` → `1-5.`

The rule preserves punctuation at the end of the date.

### Time Rules

Time rules convert various time formats to standardized representations for each language.

#### TimeEn (English)

**Rule name:** `TimeEn`

Converts time formats with colons or periods, with optional AM/PM markers.

**Examples:**
- `3:45PM!` → `3.45PM!`
- `12.30.` → `12.30.`
- `12:30` → `12.30`

#### TimeFr (French)

**Rule name:** `TimeFr`

Converts French time formats (with 'h' separator or colons).

**Examples:**
- `9h15,` → `9h15,`
- `14:00?` → `14h00?`

#### TimeDe (German)

**Rule name:** `TimeDe`

Converts German time formats (colons or periods).

**Examples:**
- `8:20.` → `8.20.`
- `22.45!` → `22.45!`

### Number Rules

Number rules expand large numbers into word-based representations for better pronunciation. Years (1900-2100) and small numbers (< 1000) are kept as-is.

**Rule names:** `NumberEn`, `NumberFr`, `NumberDe`, `NumberEs`, `NumberPt`

**English examples:**
- `123` → `123` (small numbers unchanged)
- `1234` → `1 thousand 234`
- `1000000` → `1 million`
- `2500000` → `2 million 500 thousand`
- `1002003004` → `1 billion 2 million 3 thousand 4`
- `-4500` → `minus 4 thousand 500`

**French examples:**
- `1234` → `mille 234` (singular form for 1)
- `2234` → `2 mille 234`
- `2000000` → `2 millions`
- `-4500` → `moins 4 mille 500`
- `123456000789` → `123 milliards 456 millions 789`

**Language-specific separators:**
- **English:** thousand, million, billion
- **French:** mille, million(s), milliard(s)
- **German:** Tausend, Million(en), Milliarde(n)
- **Spanish:** mil, millón/millones, mil millones
- **Portuguese:** mil, milhão/milhões, bilhão/bilhões

### Email Rules

Email rules spell out email addresses with language-specific words for special characters.

**Rule names:** `EmailEn`, `EmailFr`, `EmailDe`, `EmailEs`, `EmailPt`

**English examples:**
- `foo.bar@gmail.com` → `foo dot bar at gmail dot com`

**French examples:**
- `foo@gmail.com` → `foo arobaze gmail point com`

**Special character translations:**
- `@` → "at" (en), "arobaze" (fr), "at" (de), "arroba" (es), "arroba" (pt)
- `.` → "dot" (en), "point" (fr), "Punkt" (de), "punto" (es), "ponto" (pt)
- `-` → "dash" (en), "tiret" (fr), "Bindestrich" (de), "guión" (es), "hífen" (pt)

### URL Rules

URL rules spell out URLs including protocol, domain, path, and special characters.

**Rule names:** `UrlEn`, `UrlFr`, `UrlDe`, `UrlEs`, `UrlPt`

**English examples:**
- `www.example.com` → `www dot example dot com`
- `https://www.example.com/path` → `H-T-T-P-S colon slash slash www dot example dot com slash path`
- `http://sub.domain.co.uk` → `H-T-T-P colon slash slash sub dot domain dot C-O dot U-K`

**French examples:**
- `https://www.kyutai.fr` → `H-T-T-P-S deux-points slash slash www point kyutai point F-R`
- `www.it-management.com/promo` → `www point I-T tiret management point com slash promo`

Two-letter top-level domains are spelled out (e.g., "UK" → "U-K", "FR" → "F-R").

### Phone Number Rules

Phone number rules format phone numbers according to country-specific conventions.

**Rule names:** `PhoneEn`, `PhoneFr`, `PhoneDe`, `PhoneEs`, `PhonePt`

Phone numbers can be:
- **International format:** Starting with `+` and a country code
- **Local format:** Starting with `0`

**French examples:**
- `0123456789` → `01 23 45 67 89`
- `+330556791936` → `+33 05 56 79 19 36` (French TTS)
- `+330556791936` → `+33 0-5 5-6 7-9 1-9 3-6` (English TTS)

**English examples:**
- `07596854413` → `0-7-5-9 6-8-5 4-4-1-3`
- `+16502349653` → `+1 6-5-0 2-3-4 9-6-5-3`
- `+447700900123` → `+44 7-7-0 0-9-0 0-1-2-3` (mobile)
- `+442000900123` → `+44 2-0 0-0-9-0 0-1-2-3` (London)

**German examples:**
- `01511234567` → `0-1-5 1 1 2-3 4-5 6-7`
- `+491511234567` → `+49 1-5 1 1 2-3 4-5 6-7`

**Supported country codes:**
- `+1` - North America
- `+33` - France
- `+34` - Spain
- `+44` - United Kingdom
- `+49` - Germany
- `+351` - Portugal

### AlNum (Alphanumeric)

**Rule name:** `AlNum`

Handles mixed uppercase letters and digits (e.g., license plates, product codes).

**Examples:**
- `AB12CD34!` → `A-B 1-2 C-D 3-4!`

Characters are grouped by type (letters vs. digits) and joined with hyphens within each group.

## Best Practices

1. **Use language aliases** when possible for comprehensive coverage in a single language
2. **Combine specific rules** when you need fine-grained control or multi-language support
3. **Preserve punctuation** - rules preserve trailing punctuation (periods, commas, etc.)
4. **International phone numbers** require at least 6 digits to be recognized
5. **Year detection** - numbers between 1900-2100 are kept as-is and not expanded

## Implementation Notes

- Rules are applied word-by-word to the input text
- Only the first matching rule is applied to each word
- Special characters like quotes, dashes, and brackets are normalized before processing
- Colons (`:`) are handled specially to support time and URL formats
- When no rules are specified, minimal text normalization is applied
