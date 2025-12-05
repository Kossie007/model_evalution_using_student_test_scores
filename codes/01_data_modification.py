# importing libraries

import importlib
import subprocess
import sys

# if the package is downloaded load it, if not download it
def install_and_import(package, import_name=None):
    if import_name is None:
        import_name = package

    try:
        return importlib.import_module(import_name)
    except ImportError:
        print(f"{import_name} not found, installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return importlib.import_module(import_name)


pd = install_and_import("pandas")
np = install_and_import("numpy")
plt = install_and_import("matplotlib.pyplot", "matplotlib.pyplot")

# working directory check
# %pwd

# data read
data = pd.read_csv('../data/okm_diak_adat.csv', low_memory = False)
data.head()

data.info()

# renaming some coloumns, based on the provided file description 
data.rename(columns={
    'Unnamed: 0': 'row_numbering',
    'X.x': 'id_of_students',
    'X.y': 'id_of_schools'
}, inplace=True)

# correlation between our two independent variables
print("Correlation between maths_8 points and reading_8 points:", data['matek_8'].corr(data['szovegert_8']))
print("Correlation between maths_8 points and maths_6 points:", data['matek_8'].corr(data['matek_6']))
print("Correlation between reading_8 points and reading_6 points:", data['szovegert_8'].corr(data['szovegert_6']))

# there are lot of variables with missing values
# threshold for the proportion of missing values above which we remove the variable
na_ratio=0.1

# theoretical NA proportion above which dropping a variable may be worth considering
percentage_of_NA=0.05


# data read once again so rerunning this chunk with different NA ratio wont influence the database
data = pd.read_csv('../data/okm_diak_adat.csv', low_memory = False)
data.rename(columns={
    'Unnamed: 0': 'row_numbering',
    'X.x': 'id_of_students',
    'X.y': 'id_of_schools'
}, inplace=True)

# omitting all NAs in our independent variables
data = data.dropna(subset=['matek_8',"szovegert_8"])

# dropingp columns from df where the proportion of missing values (NaN) is higher than 'ratio' (default: 0.60 = 60%).
def remove_highnan(df,ratio):
  for i in df.columns:
    if df[i].isnull().mean() > ratio:
      df.drop(i,axis = 1,inplace = True) # if its more than 0.6 ratio of Nas, drop the column

remove_highnan(data, na_ratio)

print(f"{percentage_of_NA:.0%} of the dataframe length (red line):", round(len(data)*percentage_of_NA,1), "(just for help deciding which variables to use)")
print()
data.isnull().sum().plot(kind = 'bar', fontsize = 12,figsize = (12,4))
plt.title(f'Variables less than {na_ratio:.0%} NA ratio', fontsize=18)
plt.axhline(y=len(data)*percentage_of_NA, color='red')
plt.show()

print("Available features after filtering NAs:")
print("-----")
for col in data.columns:
    print(col)
print("-----")    

# filtering for duplicates and not useful features
# are the 2 number of teacher features equal?
tmp = data.dropna(subset=["tanarok_szama", "tanarok_szama_telephelyen"])
print(len(tmp["tanarok_szama"] == tmp["tanarok_szama_telephelyen"]))
print(sum(tmp["tanarok_szama"] == tmp["tanarok_szama_telephelyen"]))
# all rows are the same, then drop one

# are the 2 number of student features equal?
tmp = data.dropna(subset=["diak_osszletszam", "diak_osszletszama_telephelyen"])
print(len(tmp["diak_osszletszam"] == tmp["diak_osszletszama_telephelyen"]))
print(sum(tmp["diak_osszletszam"] == tmp["diak_osszletszama_telephelyen"]))
# all rows are the same, then drop one

ata.drop(columns=["diak_osszletszama_telephelyen",
                   "tanarok_szama_telephelyen"],
          inplace=True)


# renaming the coloumns
data.rename(columns={
    # already partly English but make them consistent
    'row_numbering': 'row_number',
    'id_of_students': 'student_id',
    'id_of_schools': 'school_id',

    # IDs, structure
    'telephely_azonosito': 'site_id_(telephely)',      # school site identifier (campus/site)
    'sorszam_diak': 'student_serial_number',           # internal student serial number
    'osztid': 'class_id',
    'oszt_letszam': 'class_size',

    # teacher & student counts
    'tanarok_szama': 'teachers_number_site',           # number of teachers at the site
    'diak_osszletszam': 'total_students_site',         # total number of students at the site

    # geography / school context
    'mkod_th': 'county_code_site',                     # county code at site level
    'megye_kodja': 'county_code_school',               # county code at school level
    'teltip7_th': 'settlement_type_site',              # settlement type of the site
    'telepulestipus': 'settlement_type_school',        # settlement type of the school
    'ft_csop': 'maintainer_type_group',                # maintainer type group (code)
    'fenntarto_tipuscsoport': 'maintainer_type_group_school',  # maintainer type group at school level
    'tipus': 'school_type',                            # type of school

    # test scores (standardized)
    'matek_8': 'math_score_8_std',                     # standardized math score, grade 8
    'szovegert_8': 'reading_score_8_std',              # standardized reading score, grade 8
    'matek_6': 'math_score_6_std',                     # standardized math score, grade 6
    'szovegert_6': 'reading_score_6_std',              # standardized reading score, grade 6

    # family background
    't28': 'mother_education_level',                   # mother’s or female guardian’s highest education
    't29': 'father_education_level',                   # father’s or male guardian’s highest education
    't36': 'books_at_home',                            # number of books at home (categories)
    'csh_index': 'family_background_index_std',        # standardized family background index

    # student-level attributes
    'sex': 'student_gender',                           # student gender (coded)
    't13': 'class_curriculum_type',                    # type of curriculum of the student’s class
    'hhh': 'multiplied_disadvantaged'                    # student is multiply disadvantaged (binary)
}, inplace=True)

data.head()

# filtering again for duplicates and not useful features
# we dont need these (dont have useful information )
print(data['row_number'].nunique())

print(data['student_id'].nunique())
print(data['student_serial_number'].nunique())

print(data['school_id'].nunique())
print(data['site_id_(telephely)'].nunique())

print(data['county_code_site'].nunique())
print(data['county_code_school'].nunique())

print(data['settlement_type_site'].nunique())
print(data['settlement_type_school'].nunique())

print(data['maintainer_type_group'].nunique())
print(data['maintainer_type_group_school'].nunique())

print(data['school_type'].nunique())  # no useful information

print(data['class_id'].nunique()) # also just an unnecessary feature

# removing the unnecessary features
cols_to_drop = [
    'row_number',            # row number 
    'student_id',            # student_id
    'student_serial_number', # student_serial_number
    'school_id',             # school_id
    'county_code_site',      # county_code at site level
    'settlement_type_site',  # settlement_type_site
    'maintainer_type_group', # maintainer_type_group
    'school_type',           # school_type
    'class_id',               # class_id
    'reading_score_8_std',   # we dont use these scores, since we focusing on 8th grade maths
    'math_score_6_std',                   
    'reading_score_6_std', 
]

data = data.drop(columns=cols_to_drop, errors='ignore')
data.isnull().sum().plot(kind = 'bar', fontsize = 12,figsize = (12,4))
plt.title(f'Variables less than {na_ratio:.0%} NA ratio', fontsize=18)
plt.axhline(y=len(data)*percentage_of_NA, color='red')
plt.show()

# saving the plot
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12   # base font size

data.isnull().sum().plot(kind='bar', fontsize=12, figsize=(12, 4))

plt.title(f'Variables less than {na_ratio:.0%} NA ratio', fontsize=18)

# extra comment inside the plot
plt.text(
    x=-0.3,
    y=len(data)*percentage_of_NA + 1650,
    s='Blue bars = number of NAs',
    ha='left',
    fontsize=12,              # comment font size
    fontfamily='Times New Roman'
)

# saving
plt.savefig('../figures/na_plot.png', dpi=300, bbox_inches='tight')
plt.close()


data.head()

# dropping NAs
data = data.dropna()

# manual type specification for each variable
dtype_spec = {
    "site_id_telephely":          "category",  # school site ID (identifier)
    "class_size":                 "int",       # number of students in the class
    "teachers_number_site":       "float",       # number of teachers at the site
    "math_score_8_std":           "float",     # standardized math score (grade 8)
    "total_students_site":        "int",       # total number of students at the site
    "mother_education_level":     "category",  # coded education level -> categorical
    "father_education_level":     "category",
    "books_at_home":              "category",  # coded categories (0–50, 50–150, etc.)
    "family_background_index_std":"float",     # continuous index
    "student_gender":             "category",  # boy/girl
    "class_curriculum_type":      "category",  # normal / bilingual / specialized / nationality
    "multiply_disadvantaged":     "category",  # yes / no (binary factor)
    "county_code_school":         "category",  # region code – categorical, not numeric scale
    "settlement_type_school":     "category",  # Budapest / town / village, etc.
    "maintainer_type_group_school":"category"  # state / church / private, etc.
}

for col, kind in dtype_spec.items():
    if col not in data.columns:
        print(f"Warning: column '{col}' not found in data")
        continue

    if kind == "category":
        data[col] = data[col].astype("category")

    elif kind == "int":
        # convert to numeric then to nullable integer
        data[col] = pd.to_numeric(data[col], errors="coerce").astype("Int64")

    elif kind == "float":
        data[col] = pd.to_numeric(data[col], errors="coerce")

# quick check
print(data.dtypes)

data["books_at_home"].unique() # check is there is NA -> no NA


edu_map = {
    1: "less_than_primary",       # kevesebb mint 8 általános
    2: "primary_school",          # általános iskola
    3: "vocational_school",       # szakiskola
    4: "apprenticeship_school",   # szakmunkásképző
    5: "secondary_with_matura",   # érettségi
    6: "college",                 # főiskola
    7: "university"               # egyetem
}

books_map = {
    1: "0_50",                    # kb 0–50
    2: "around_50",               # kb 50
    3: "up_to_150",               # max 150
    4: "up_to_300",               # max 300
    5: "300_600",                 # 300–600
    6: "600_1000",                # 600–1000
    7: "more_than_1000"           # 1000-nél több
}

gender_map = {
    0: "boy",                     # fiú
    1: "girl"                     # lány
}

curriculum_map = {
    1: "normal",                  # normál
    2: "bilingual",               # két tannyelvű
    3: "specialized",             # tagozatos
    6: "nationality"              # nemzetiségi
}

hhh_map = {
    0: "no",                      # nem
    1: "yes"                      # igen
}

settlement_map = {
    1: "Budapest",
    2: "county_seat",
    3: "town",
    4: "village_under_5000",
    5: "village_2000_5000",
    6: "village_1000_2000",
    7: "village_under_1000"
}

maintainer_map = {
    2: "local_municipality",      # települési/kerületi önkormányzat
    3: "central_government_state",# központi költségvetés/állami
    4: "church",                  # egyházi
    5: "foundation_private",      # alapítvány/magán
    6: "other"                    # egyéb
}

# --- apply maps --- 

# mother & father education
for col in ["mother_education_level", "father_education_level"]:
    if col in data.columns:
        tmp = pd.to_numeric(data[col], errors="coerce")
        data[col] = tmp.map(edu_map).astype("category")

# books at home
if "books_at_home" in data.columns:
    tmp = pd.to_numeric(data["books_at_home"], errors="coerce")
    data["books_at_home"] = tmp.map(books_map).astype("category")

# student gender
if "student_gender" in data.columns:
    tmp = pd.to_numeric(data["student_gender"], errors="coerce")
    data["student_gender"] = tmp.map(gender_map).astype("category")

# class curriculum type
if "class_curriculum_type" in data.columns:
    tmp = pd.to_numeric(data["class_curriculum_type"], errors="coerce")
    data["class_curriculum_type"] = tmp.map(curriculum_map).astype("category")

# multiplied_disadvantaged (hhh) – currently int64
if "multiplied_disadvantaged" in data.columns:
    tmp = pd.to_numeric(data["multiplied_disadvantaged"], errors="coerce")
    data["multiplied_disadvantaged"] = tmp.map(hhh_map).astype("category")

# settlement type of school
if "settlement_type_school" in data.columns:
    tmp = pd.to_numeric(data["settlement_type_school"], errors="coerce")
    data["settlement_type_school"] = tmp.map(settlement_map).astype("category")

# maintainer type group at school level
if "maintainer_type_group_school" in data.columns:
    tmp = pd.to_numeric(data["maintainer_type_group_school"], errors="coerce")
    data["maintainer_type_group_school"] = tmp.map(maintainer_map).astype("category")

# county_code_school: keep as categorical codes (no full county-name map in codebook snippet)
if "county_code_school" in data.columns:
    data["county_code_school"] = data["county_code_school"].astype("category")
	

for c in ["mother_education_level", "father_education_level",
          "books_at_home", "student_gender",
          "class_curriculum_type", "multiplied_disadvantaged",
          "settlement_type_school", "maintainer_type_group_school"]:
    if c in data.columns:
        print("\n", c)
        print(data[c].value_counts(dropna=False))
		
# descriptive stats for numerical features 
# select only numeric columns
num_data = data.select_dtypes(include=[np.number])

# get descriptive statistics
desc = num_data.describe().T.round(2)  # transpose so variables are rows

print(desc)

# filter those errors where the teacher and student number are zero
data = data[~((data['total_students_site'] == 0) |
              (data['teachers_number_site'] == 0))]
			  
len(data) # 56590 -> 56569 21 obs

data.isnull().sum().plot(kind = 'bar', fontsize = 12,figsize = (12,4))
plt.title(f'Variables less than {na_ratio:.0%} NA ratio', fontsize=18)
plt.show()

# creating a student/teacher ratio, because we think that is more informative
data["student_teacher_ratio"] = data["total_students_site"] / data["teachers_number_site"]

# drop the original columns
data = data.drop(columns=["total_students_site", "teachers_number_site"])


data.info()

data.head()

seed = 314+133+76223+3+5
np.random.seed(seed)

# 90% random sample for analysis
filtered_data_anal = data.sample(frac=0.9, random_state=seed)

# remaining 10% for final evaluation
filtered_data_eval = data.drop(filtered_data_anal.index)

# write to CSVs
filtered_data_anal.to_csv('../data/filtered_data_anal.csv', index=False)
filtered_data_eval.to_csv('../data/filtered_data_eval.csv', index=False)