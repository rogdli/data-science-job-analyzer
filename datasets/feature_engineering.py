import pandas as pd

class FeatureEngineer:
    """
    >>> from feature_engineering import FeatureEngineer
    >>> fe = FeatureEngineer(df_features)
    >>> df_ready = fe.engineer_all()
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.state_city_to_country = self._init_state_city_to_country()
        self.location_to_continent = self._init_location_to_continent()

    # -------------------------
    # Utilities
    # -------------------------
    @staticmethod
    def words_to_num(text: str) -> str:
        transformer_dict = {
            "zero": "0", "a ": "1 ", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
            "ten": "10", "eleven": "11", "twelve": "12",
        }
        for word, num in transformer_dict.items():
            text = text.replace(word, num)
        return text

    @staticmethod
    def convert_revenue_to_numeric(revenue_str):
        if isinstance(revenue_str, str):
            revenue_str = revenue_str.replace('€', '').replace('B', 'e9').replace('M', 'e6').replace(',', '')
            try:
                return float(revenue_str)
            except ValueError:
                return None
        return None

    # -------------------------
    # Post date
    # -------------------------
    def extract_post_date(self, el) -> int:
        el = self.words_to_num(el)
        number = el.split()[0]
        if 'hour' in el.lower():
            days = 0
        elif 'day' in el.lower():
            days = int(number)
        elif 'month' in el.lower():
            days = int(number) * 30
        elif 'year' in el.lower():
            days = int(number) * 365
        else:
            raise ValueError(f'Unexpected format in post_date: {el}')
        return days

    def engineer_post_date(self):
        self.df['post_date'] = self.df['post_date'].apply(self.extract_post_date)
        self.df = self.df[self.df['post_date'] <= 360]

    # -------------------------
    # Company size, ownership & revenue
    # -------------------------
    def engineer_company_info(self):
        ownerships = self.df['ownership'].dropna().unique()

        # step 1: ownership <- revenue
        mask = self.df['revenue'].isin(ownerships)
        self.df.loc[mask, 'ownership'] = self.df.loc[mask, 'revenue']

        # step 2: ownership <- company_size
        mask = self.df['company_size'].isin(ownerships)
        self.df.loc[mask, 'ownership'] = self.df.loc[mask, 'company_size']

        # step 3: revenue <- company_size if contains €
        mask = self.df['company_size'].str.contains('€', na=False)
        self.df.loc[mask, 'revenue'] = self.df.loc[mask, 'company_size']

        # step 4: drop company_size if invalid
        mask = self.df['company_size'].str.contains('€', na=False) | self.df['company_size'].isin(ownerships)
        self.df.loc[mask, 'company_size'] = None

        # step 5: keep revenue only if valid
        mask = self.df['revenue'].astype(str).str.contains('€', na=False)
        self.df.loc[~mask, 'revenue'] = None

        # step 6: convert revenue
        self.df['revenue'] = self.df['revenue'].apply(self.convert_revenue_to_numeric)

        # step 7: clean company_size
        self.df['company_size'] = self.df['company_size'].apply(
            lambda x: int(''.join(x.split(','))) if isinstance(x, str) else None
        )

    # -------------------------
    # Location & Headquarter
    # -------------------------
    def _init_state_city_to_country(self):
        return {'AL': 'United States', 'AK': 'United States', 'AZ': 'United States', 'AR': 'United States', 'CA': 'United States',
                'CO': 'United States', 'CT': 'United States', 'DE': 'United States', 'FL': 'United States', 'GA': 'United States',
                'HI': 'United States', 'ID': 'United States', 'IL': 'United States', 'IN': 'United States', 'IA': 'United States',
                'KS': 'United States', 'KY': 'United States', 'LA': 'United States', 'ME': 'United States', 'MD': 'United States',
                'MA': 'United States', 'MI': 'United States', 'MN': 'United States', 'MS': 'United States', 'MO': 'United States',
                'MT': 'United States', 'NE': 'United States', 'NV': 'United States', 'NH': 'United States', 'NJ': 'United States',
                'NM': 'United States', 'NY': 'United States', 'NC': 'United States', 'ND': 'United States', 'OH': 'United States',
                'OK': 'United States', 'OR': 'United States', 'PA': 'United States', 'RI': 'United States', 'SC': 'United States',
                'SD': 'United States', 'TN': 'United States', 'TX': 'United States', 'UT': 'United States', 'VT': 'United States',
                'VA': 'United States', 'WA': 'United States', 'WV': 'United States', 'WI': 'United States', 'WY': 'United States',
                'DC': 'United States',
                'USA': 'United States', 'United States': 'United States', 'US': 'United States',

                'ON': 'Canada', # Ontario
                'QC': 'Canada', # Quebec
                'AB': 'Canada', # Alberta
                'BC': 'Canada', # British Columbia
                'MB': 'Canada', # Manitoba
                'SK': 'Canada', # Saskatchewan
                'NB': 'Canada', # New Brunswick
                'NL': 'Canada', # Newfoundland and Labrador
                'NS': 'Canada', # Nova Scotia
                'PE': 'Canada', # Prince Edward Island
                'Canada': 'Canada',

                'Karnataka': 'India',
                'Maharashtra': 'India',
                'India': 'India',

                'SP': 'Brazil', # Sao Paulo
                'Brazil': 'Brazil',

                'BS': 'Switzerland', # Basel
                'ZH': 'Switzerland', # Zurich
                'Switzerland': 'Switzerland',

                'BW': 'Germany', # Baden-Württemberg
                'HE': 'Germany', # Hesse
                'BY': 'Germany', # Bavaria
                'Germany': 'Germany',

                'LI': 'Netherlands', # Limburg
                'NB': 'Netherlands', # North Brabant
                'Netherlands': 'Netherlands',

                'CT': 'Spain', # Catalonia
                'Spain': 'Spain',

                'UK': 'United Kingdom', 'United Kingdom': 'United Kingdom', 'GB': 'United Kingdom',
                'Ireland': 'Ireland',
                'Singapore': 'Singapore', 'SG': 'Singapore',
                'Mexico': 'Mexico',
                'Poland': 'Poland',
                'Romania': 'Romania',
                'Lithuania': 'Lithuania',
                'Estonia': 'Estonia',
                'Bulgaria': 'Bulgaria',
                'Portugal': 'Portugal',
                'Austria': 'Austria', 'AT': 'Austria',
                'Argentina': 'Argentina',
                'China': 'China', 'CN': 'China',
                'Denmark': 'Denmark', 'DK': 'Denmark',
                'Sweden': 'Sweden', 'SE': 'Sweden',
                'Norway': 'Norway',
                'Finland': 'Finland',
                'Belgium': 'Belgium',
                'New Zealand': 'New Zealand',
                'South Korea': 'South Korea',
                'Taiwan': 'Asia', 'TW': 'Asia',
                'Hong Kong': 'Asia',
                'South Africa': 'South Africa',
                'Japan': 'Asia', 'JP': 'Asia',
                'Australia': 'Australia', 'AU': 'Australia',

                # Add cities and their countries
                'London': 'United Kingdom',
                'New York': 'United States',
                'San Francisco': 'United States',
                'Bengaluru': 'India',
                'Tokyo': 'Asia',
                'Paris': 'France',
                'Sydney': 'Australia',
                'Singapore': 'Singapore',
                'Mexico City': 'Mexico',
                'Sao Paulo': 'Brazil',
                'Warsaw': 'Poland',
                'Bucharest': 'Romania',
                'Vilnius': 'Lithuania',
                'Tallinn': 'Estonia',
                'Sofia': 'Bulgaria',
                'Lisbon': 'Portugal',
                'Vienna': 'Austria',
                'Buenos Aires': 'Argentina',
                'Zurich': 'Switzerland',
                'Basel': 'Switzerland',
                'Berlin': 'Germany',
                'Frankfurt': 'Germany', 'Frankfurt am Main': 'Germany',
                'Amsterdam': 'Netherlands',
                'Dublin': 'Ireland', 'Dublin 2': 'Ireland',
                'Barcelona': 'Spain',
                'Madrid': 'Spain',
                'Copenhagen': 'Denmark', 'København': 'Denmark',
                'Stockholm': 'Sweden',
                'Oslo': 'Norway',
                'Helsinki': 'Finland',
                'Brussels': 'Belgium',
                'Auckland': 'New Zealand',
                'Seoul': 'South Korea',
                'Taipei': 'Asia',
                'Hong Kong': 'Asia',
                'Johannesburg': 'South Africa',
                'Mumbai': 'India',
                'Chennai': 'India',
                'Hyderabad': 'India',
                'Pune': 'India',
                'Gurugram': 'India',
                'Noida': 'India',
                'Toronto': 'Canada',
                'Vancouver': 'Canada',
                'Montreal': 'Canada', 'Montréal': 'Canada',
                'Calgary': 'Canada',
                'Seattle': 'United States',
                'Mountain View': 'United States',
                'Sunnyvale': 'United States',
                'Los Angeles': 'United States',
                'Chicago': 'United States',
                'Boston': 'United States',
                'Austin': 'United States',
                'Dallas': 'United States',
                'Houston': 'United States',
                'Philadelphia': 'United States',
                'Phoenix': 'United States',
                'San Diego': 'United States',
                'Denver': 'United States',
                'Miami': 'United States',
                'Atlanta': 'United States',
                'Detroit': 'United States',
                'Minneapolis': 'United States',
                'St. Louis': 'United States',
                'Portland': 'United States',
                'San Antonio': 'United States',
                'Charlotte': 'United States',
                'Raleigh': 'United States',
                'Nashville': 'United States',
                'Kansas City': 'United States',
                'Las Vegas': 'United States',
                'Orlando': 'United States',
                'Tampa': 'United States',
                'Cleveland': 'United States',
                'Cincinnati': 'United States',
                'Pittsburgh': 'United States',
                'Indianapolis': 'United States',
                'Columbus': 'United States',
                'Milwaukee': 'United States',
                'Virginia Beach': 'United States',
                'Providence': 'United States',
                'Salt Lake City': 'United States',
                'New Orleans': 'United States',
                'Baltimore': 'United States',
                'Louisville': 'United States',
                'Richmond': 'United States',
                'Jacksonville': 'United States',
                'Oklahoma City': 'United States',
                'Memphis': 'United States',
                'Columbia': 'United States',
                'Charleston': 'United States',
                'Des Moines': 'United States',
                'Omaha': 'United States',
                'Boise': 'United States',
                'Fargo': 'United States',
                'Cheyenne': 'United States',
                'Helena': 'United States',
                'Anchorage': 'United States',
                'Honolulu': 'United States',
                'Dover': 'United States',
                'Augusta': 'United States',
                'Montpelier': 'United States',
                'Concord': 'United States',
                'Annapolis': 'United States',
                'Sacramento': 'United States',
                'Springfield': 'United States',
                'Tallahassee': 'United States',
                'Boise': 'United States',
                'Topeka': 'United States',
                'Baton Rouge': 'United States',
                'Jackson': 'United States',
                'Jefferson City': 'United States',
                'Lincoln': 'United States',
                'Carson City': 'United States',
                'Santa Fe': 'United States',
                'Bismarck': 'United States',
                'Columbus': 'United States',
                'Oklahoma City': 'United States',
                'Salem': 'United States',
                'Harrisburg': 'United States',
                'Columbia': 'United States',
                'Pierre': 'United States',
                'Nashville': 'United States',
                'Austin': 'United States',
                'Salt Lake City': 'United States',
                'Montpelier': 'United States',
                'Richmond': 'United States',
                'Charleston': 'United States',
                'Madison': 'United States',
                'Cheyenne': 'United States',
                'Washington': 'United States',
                'Texas': 'United States',
                'Florida': 'United States',
                'California': 'United States',

                # Add other cities and their countries as needed
                'Fully Remote': 'Remote',
                'On-site': 'On-site',
                'Hybrid': 'Hybrid',

                'Alameda': 'United States',
                'Altrincham': 'United Kingdom',
                'Ann Arbor': 'United States',
                'Annapolis Junction': 'United States',
                'Arlington': 'United States',
                'Armonk': 'United States',
                'Bagsværd': 'Denmark',
                'Basking Ridge': 'United States',
                'Beaverton': 'United States',
                'Bedford': 'United States',
                'Bellevue': 'United States',
                'Benbrook': 'United States',
                'Bentonville': 'United States',
                'Berkeley': 'United States',
                'Bethesda': 'United States',
                'Birmingham': 'United States',
                'Bloomington': 'United States',
                'Boulder': 'United States',
                'Bowie': 'United States',
                'Bozeman': 'United States',
                'Brentford': 'United Kingdom',
                'Brentwood': 'United States',
                'Bridgewater': 'United States',
                'Buffalo': 'United States',
                'Buffalo Grove': 'United States',
                'Burlingame': 'United States',
                'Cambridge': 'United States',
                'Cary': 'United States',
                'Catonsville': 'United States',
                'Centreville': 'United States',
                'Cherry Hill': 'United States',
                'Chevy Chase': 'United States',
                'Collingwood': 'Australia',
                'Coraopolis': 'United States',
                'Culver City': 'United States',
                'Dania Beach': 'United States',
                'Dartmouth': 'Canada',
                'Dearborn': 'United States',
                'Deerfield': 'United States',
                'Draper': 'United States',
                'Eden Prairie': 'United States',
                'Edinburgh': 'United Kingdom',
                'Edison': 'United States',
                'El Segundo': 'United States',
                'Englewood': 'United States',
                'Evanston': 'United States',
                'Fairfax': 'United States',
                'Falls Church': 'United States',
                'Findlay': 'United States',
                'Fort Mill': 'United States',
                'Foster City': 'United States',
                'Fulton': 'United States',
                'Fuschl am See': 'Austria',
                'Gainesville': 'United States',
                'Gaithersburg': 'United States',
                'Golden Valley': 'United States',
                'Grand Rapids': 'United States',
                'Green Bay': 'United States',
                'Greenbelt': 'United States',
                'Hartford': 'United States',
                'Heerlen': 'Netherlands',
                'Herndon': 'United States',
                'Hook': 'United Kingdom',
                'Huntsville': 'United States',
                'Ingelheim am Rhein': 'Germany',
                'Irvine': 'United States',
                'Irving': 'United States',
                'Kenilworth': 'United States',
                'Kent': 'United States',
                'Kista': 'Sweden',
                'Lake Bluff': 'United States',
                'Laurel': 'United States',
                'Los Alamos': 'United States',
                'Malvern': 'United States',
                'Manchester': 'United Kingdom',
                'Maplewood': 'United States',
                'McLean': 'United States',
                'Menlo Park': 'United States',
                'Milano': 'Italy',
                'Minato City': 'Asia',
                'Miramar': 'United States',
                'Mooresville': 'United States',
                'München': 'Germany',
                'New Brunswick': 'United States',
                'Newport Beach': 'United States',
                'Newton': 'United States',
                'Newtown': 'United States',
                'North Chicago': 'United States',
                'Northbrook': 'United States',
                'Palo Alto': 'United States',
                'Plano': 'United States',
                'Princeton': 'United States',
                'Purchase': 'United States',
                'Readington Township': 'United States',
                'Redmond': 'United States',
                'Redwood City': 'United States',
                'Renton': 'United States',
                'Reston': 'United States',
                'Riverwoods': 'United States',
                'Rochester': 'United States',
                'Rockville': 'United States',
                'Roseburg': 'United States',
                'Roseland': 'United States',
                'Rosemont': 'United States',
                'Roseville': 'United States',
                'San Carlos': 'United States',
                'San Jose': 'United States',
                'Santa Clara': 'United States',
                'Santa Monica': 'United States',
                'Scottsdale': 'United States',
                'South Bend': 'United States',
                'South San Francisco': 'United States',
                'Stamford': 'United States',
                'Sugar Land': 'United States',
                'São Paulo': 'Brazil',
                'Teaneck': 'United States',
                'Torrance': 'United States',
                'Veldhoven': 'Netherlands',
                'Ventura': 'United States',
                'Walldorf': 'Germany',
                'West Menlo Park': 'United States',
                'Westminster': 'United States',
                'Woonsocket': 'United States',
                'Yonkers': 'United States',
                'Zürich': 'Switzerland'
            }

    def _init_location_to_continent(self):
        return {'United States': 	'United States',
                         'India': 'Asia',
                         'multi-location':	'multi-location',
                         'Remote':	'Remote',
                         'Canada': 	'Canada',
                         'Unrecognized':	'Other',
                         'On-site':	'On-site',
                         'Spain':	'Europe',
                         'United Kingdom': 'Europe',
                         'Singapore':	'Asia',
                         'Hybrid': 'Hybrid',
                         'Netherlands': 'Europe',
                         'Ireland': 'Europe',
                         'Estonia': 'Europe',
                         'Denmark': 'Europe',
                         'Sweden': 'Europe',
                         'Taiwan': 'Asia',
                         'China': 'Asia',
                         'Asia': 'Asia',
                         'Germany': 'Europe',
                         'Brazil': 'Other',
                         'Switzerland': 'Europe',
                         'France': 'Europe',
                         'Italy': 'Europe',
                         'Mexico': 'Other',
                         'Austria':	'Europe',
                         'Australia': 'Other'}

    @staticmethod
    def extract_location(el):
        if not pd.isna(el):
            parts = el.strip().split('.')
            if len(parts) > 2:
                return 'multi-location'
            else:
                return parts[0].split(',')[-1].strip()
        return None

    @staticmethod
    def extract_headquarter(el):
        if not pd.isna(el):
            parts = el.strip().split('.')
            if len(parts) > 2:
                return 'multi-location'
            else:
                return parts[0].split(',')[0].strip()
        return None

    def replace_location_with_country(self, location):
        if '(' in str(location):
            location = location.split('(')[0].strip()
        if not pd.isna(location):
            found = False
            for item in location.split():
                if item in self.state_city_to_country.keys():
                    found = True
                    break
            if location in self.state_city_to_country.keys():
                return self.state_city_to_country[location]
            elif 'multi-location' in location.lower():
                return 'multi-location'
            elif found:
                return self.state_city_to_country[item]
            else:
                return 'Unrecognized'
        return None

    def replace_location_with_continent(self, location):
        if not pd.isna(location):
            if location in self.location_to_continent.keys():
                return self.location_to_continent[location]
            else:
                raise ValueError(f'Unrecognized location: {location}')
        return None

    def engineer_location(self):
        # 1. Extract location
        self.df['test'] = self.df['location'].apply(self.extract_location)

        # 2. Replace On-site / Hybrid with headquarter (your explicit step)
        mask = self.df['test'].isin(['On-site', 'Hybrid'])
        self.df.loc[mask, 'test'] = self.df.loc[mask, 'headquarter']

        # 3. Map to country
        self.df['test'] = self.df['test'].apply(self.replace_location_with_country)

        # 4. Map to continent
        self.df['test'] = self.df['test'].apply(self.replace_location_with_continent)

        # 5. Assign back
        self.df['location'] = self.df['test']

        # headquarter separately (not merged into location!)
        self.df['test'] = self.df['headquarter'].apply(self.extract_headquarter)
        self.df['test'] = self.df['test'].apply(self.replace_location_with_country)
        self.df['test'] = self.df['test'].apply(self.replace_location_with_continent)
        self.df['headquarter'] = self.df['test']

    # -------------------------
    # Salary
    # -------------------------
    def engineer_salary(self):
        self.df[['min_salary', 'max_salary']] = self.df['salary'].str.extract(
            r'€([\d,\.]+) - €([\d,\.]+)'
        )
        single_salary_mask = self.df['salary'].str.match(r'^€[\d,\.]+$')
        self.df.loc[single_salary_mask, 'min_salary'] = self.df.loc[single_salary_mask, 'salary'].str.extract(
            r'€([\d,\.]+)'
        )[0].str.replace(',', '').astype(float)
        self.df.loc[single_salary_mask, 'max_salary'] = self.df.loc[single_salary_mask, 'min_salary']

        self.df['min_salary'] = self.df['min_salary'].astype(str).str.replace(',', '').astype(float)
        self.df['max_salary'] = self.df['max_salary'].astype(str).str.replace(',', '').astype(float)
        self.df['mean_salary'] = (self.df['min_salary'] + self.df['max_salary']) / 2

    # -------------------------
    # Full pipeline
    # -------------------------
    def engineer_all(self):
        self.engineer_post_date()
        self.engineer_company_info()
        self.engineer_location()
        self.engineer_salary()
        self.df.drop(['test', 'post_date', 'salary'], axis=1, inplace=True)
        return self.df

