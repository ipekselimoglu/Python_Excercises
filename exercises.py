import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

### SET OPERATIONS
kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", 'python', "miull"])

type(kume1)
kume2
kume1.issuperset(kume2)
kume2.difference(kume1)  # fark
kume1.symmetric_difference(kume2)  # ortak olmayanlar
kume1.intersection(kume2)  # kesişim
kume1.isdisjoint(kume2)  # kesişim boş mu?
kume1.issubset(kume2)  # alt kümesi mi?

null_list = []
salary = [10, 20, 30, 40, 50]

for i in salary:
    null_list.append(i)

null_list
####LIST COMPHERENSIONS
[i + 2 if i == 20 else i * 50 for i in salary]
[i * 2 for i in salary]  # diziyi gezer i*2 fonk uygular hepsine
[i * 2 for i in salary if i < 50]  # diziyi kurala göre gezip fonk.u uygular , tek başına if kull. en sağda
[i * 2 if i < 50 else i * 10 for i in salary]  # eğer if-else kullanılacaksa for en sağda olur

########
st = ['John', 'Mark', 'Vanessa', 'Mariam']
st_no = ['John', 'Vanessa']
type(st)

[i.lower() if i in st else i.upper() for i in st]

[i.lower() if i in st_no else i.upper() for i in st]
[i.lower() if i not in st_no else i.upper() for i in st]

########
dictionary = {"a": 1, "b": 2, "c": 3}
dictionary.keys()
dictionary.items()
dictionary.values()

{k: v ** 2 for (k, v) in dictionary.items()}
{k.upper(): v for (k, v) in dictionary.items()}

#######
numbers = range(10)
new_dict = {}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2

var = {n: n ** 3 for n in numbers if n % 2 == 0}
###
df = sns.load_dataset("car_crashes")
df.columns

a = []
for i in df.columns:
    a.append(i.upper())

a

df.columns = [i.upper() for i in df.columns]
df.columns

[i for i in df.columns if "INS" in i]

["flag_" + col for col in df.columns if "INS" in col]

["flag_" + col if "INS" in col else 'no_flag_' + col for col in df.columns]

import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns

num_cols = [col for col in df.columns if df[col].dtype != "0"]
num_cols

soz = {}
agg_list = ["mean", "min"]

for col in num_cols:  # {col:agg_list col for col in num_cols}  aynısı
    soz[col] = agg_list

soz

df.head()
new_dict = {col: agg_list for col in num_cols}
new_dict

df[num_cols].agg(new_dict)
liste = [1, 2, 3, 4, 5]
yeni_liste = [i for i in liste]
print(yeni_liste)


def AnalyseSet(set1, set2):
    if set1.issuperset(set2):
        return set1.intersection(set2)
    else:
        return set2.difference(set1)


new_set = AnalyseSet(kume1, kume2)
print(new_set)
########
from pandas.api.types import is_numeric_dtype

[f"NUM_{col.upper()}" if is_numeric_dtype(df[col]) else col.upper() for col in df.columns]  # nümerik mi kontrolü

#######
df = sns.load_dataset("car_crashes")
df.columns
type(df.columns[0])

df.columns = [col.upper() if df[col].dtype == "O" else "NUM_" + col.upper() for col in df.columns]
df.columns

df.columns = [col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]
df.columns

og_list = ["abbrev", "no_previous"]
new_col = [col for col in df.columns if col not in og_list]
new_df = df[new_col]
new_df.columns

type(df)

#####
x = "şsjldl kajla, şlask dşak, şakd lasş."
x = x.upper()
x = x.replace(",", " ")
x = x.replace(".", " ").split()  # split her bölümü ayırır
x
#####
text = "The goal is to turn data into information, and information into insight."
text = text.upper()
list_of_words = re.sub(r"[.,]", " ", text.upper()).split()

liste = ["d", "a", "t", "a", "s", "c", "i", "e", "n", "c", "e"]
len(liste)
liste[0]
liste[10]
liste[0:4]
liste.remove(liste[8])
liste.pop(8)  # liste.remove(liste[8]) ile aynı ancak hangi karakter siliniyor onu da yazdırır
liste.append("xx")
liste.insert(8, "n")
liste

dict = {"cristian": ["america", 18],
        "daisy": ["england", 12],
        "anto": ["spain", 22],
        "dante": ["italy", 25]}
dict.keys()
dict.values()
dict.items()
dict["daisy"][1] = "13"
dict.items()
dict.update({"Ahmet": ["turkey", 24]})
dict["ahmet"] = ["turkey", 24]  # dict.update({"Ahmet": ["turkey", 24]}) ile aynı
dict.items()
dict.pop("dante")
dict.items()

#######
liste = [1, 2, 3, 5, 6, 8, 9, 0, 10, 14, 16]
list1 = []
list2 = []


def func(input_lst):
    for i in range(len(liste)):
        if i % 2 == 0:
            list1.append(i)

        else:
            list2.append(i)
    return list1, list2


list1, list2 = func(liste)
print(list1)
print(list2)

#####
l = [2, 13, 18, 93, 22]


def odds_evens(numbers: list[int]) -> (list[int], list[int]):
    """Takes in a list of numbers and returns two lists => odds, evens

    Args:
        numbers: list of int numbers

    Returns:
        odds: list of odd numbers
        evens: list of even numbers

    """
    # odds = [number for number in numbers if number % 2 != 0]
    # evens = [number for number in numbers if number % 2 == 0]
    odds = []
    evens = []
    [odds.append(number) if number % 2 != 0 else evens.append(number) for number in numbers]
    return odds, evens


odds, evens = odds_evens(l)

####

ogr = ["ali", "veli", "ayşe", "talat", "zeynep", "ece"]

for i, j in enumerate(
        ogr):  # bir dizi veya herhangi bir sıralı veri yapısı üzerinde döngü yaparken hem elemanın değerini hem de o elemanın dizinini almamızıs ağlar
    if i == 0:
        print("mühendilik fak. 1. öğrenci:" + j)
    elif i == 1:
        print("mühendilik fak. 2. öğrenci:" + j)
    else:
        print("mühendilik fak. 3. öğrenci:" + j)

###
ogrs = ["ali", "veli", "ayşe", "talat", "zeynep", "ece"]


def func(ogrs):
    list1 = ogrs[0:3]
    list2 = ogrs[3:6]

    for index, ogr in enumerate(list1, 1):
        print(str(index) + ogr)
    for index, ogr in enumerate(list2, 1):
        print(str(index) + ogr)


func(ogrs)

###
ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

for i, ogrenci in enumerate(ogrenciler):  # ogrenci refers to student
    if i < 3:
        print(f"{ogrenci}, Mühendislik Fakültesi")  # "Mühendislik Fakültesi" refers to "Engineering Faculty"
    else:
        print(f"{ogrenci}, Tıp Fakültesi")  # "Tıp Fakültesi" refers to "Medical Faculty"

######

ders_kod = ["cmp1", "mn22", "lll9", "ppp2"]
kredi = [3, 4, 2, 4]
kont = [30, 75, 150, 25]

liste = list(zip(ders_kod, kredi, kont))

for i, ders in enumerate(liste):
    print("Kredisi " + str(liste[i][1]) + " olan " + liste[i][0] + " kodlu dersin kontenjanı " + str(
        liste[i][2]) + " kişidir.")

[print("Kredisi {} olan {} kodlu dersin kontenjanı {} kişidir.".format(a, b, c)) for a, b, c in
 zip(kredi, ders_kod, kont)]

####################
# NUMPY fixed type veri tutar, vektörel yüksek seviyede işlem yapar
# numpy'ın kendi array yapısı vardır ndarray
a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b

type(a)
np.zeros(10, dtype=int)  # 10 luk 0 dizisi döner
np.random.randint(0, 10, size=10)
np.random.normal(10, 4, (3, 4))  # ortalaması 10 std 4 olan 3e 4lük normal dağılımlı bir array oluşturur

a = np.random.randint(10, size=5)  # ilk argümana bişey girilmezse 0 dan alır.
a
a.ndim  # boyut sayısı
a.shape
a.size
a.dtype

a = np.random.randint(10, size=10)
a.reshape(5, 2)

a = np.random.randint(10, size=10)
a
a[0]
a[0:5]
a[0] = 999

m = np.random.randint(10, size=(3, 5))
m
m[0, 0]

m[2, 1] = 999
m
m[2, 1] = 2.1  # inte çevirip değiştirir
m
m[0, :]
m[:, 0]
# fancey index
v = np.arange(0, 30, 3)  # 0 dan 30 da 3 er artan eleman
v[1]
catch = [1, 2, 3]
v[catch]  # istenilen indexler listeden okunur
# array içinde koşullu eleman seçme
v = np.array([1, 2, 3, 4, 5])
v < 3
v > 3
v[v != 3]
v[v == 3]
v[v >= 3]
v / 5
v * 5 / 10
v ** 2
v - 1
v + 5

np.subtract(v, 1)
np.add(v, 1)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)

##3 formül çözme
# 5*x0 +x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 3]])
b = np.array([12, 10])

np.linalg.solve(a, b)

# NUMPY her verinin ayrı ayrı tipini tutmaz, tek bir tip tutar ve o tipteki verileri içinde saklar ve daha hızlıdır.!!!
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[-3:-1])

###PANDAS:
s = pd.Series([10, 22, 22, 3, 4, 4, 5, 7])  # index bilgileri de var
type(s)

s.index
s.dtype
s.size
s.ndim
s.values
type(s.values)
s.head(2)

df = pd.read_csv('C:/Users/ipeks/PycharmProjects/pythonProject/WEEK1/advertising.csv')
df.head()

import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.info()

df.columns
df.index
df.describe().T
df.isnull().values
df.isnull().values.any()  # hiç null var mı?
df.isnull().sum()
df["sex"].head()
df["sex"].value_counts()  # kategoriler ve sayıları döner !!!

df[0:15]
df.drop(0, axis=0)  # ilk satırı siler
df.drop(1, axis=0, inplace=True)  # kalıcı yapar
df

df["age"]
df.age.head()
df.index = df["age"]  # index ismi değiştirme  # age i index yapar
df
df.drop("age", axis=1, inplace=True)  # age kolonunu siler
df
df["age"] = df.index
df.head()
df.drop("age", axis=1, inplace=True)
df
df = df.reset_index().head()  # indexte yer alan değişkeni siler, ve sütun olarak ekler!!!
df
pd.set_option('display.max_columns', None)  # kolonların hepsini yazar
df

"age" in df  # age değişkeni df nin içinde var mı
df.age.head()

type(df["age"].head())  # pandas serisi olarak döner
type(df[["age"]].head())  # dataframe olarak döner ÖNEMLİ
df[["age", "deck"]]

df["age2"] = df["age"] ** 2
df
df["age3"] = df["age"] / df["age2"]

df.loc[:, df.columns.str.contains("age")].head()  # age içeren kolonları döndürür
# iloc index bilgisi vererek seçim yapma
# loc mutlak olarak indexlerdeki labellara göre seçim yapar

# iloc(integer based selection) son eleman da dahil
df.iloc[0:3]  # 3.eleman da dahil
df.iloc[0, 0]
# loc:label based selection -e kadar
df.loc[0:3]

df.iloc[0:3, "age"]  # HATALI  integer based olduğu için aralık vermek lazım
df.iloc[0:3, 0:3]
df.loc[0:3, "age"]

col_nms = ["age", "embarked", "alive"]
df.loc[0:3, col_nms]
df
df[df["age"] > 5].count()
df.loc[df["age"] > 5, "class"].head()  # yaşı 5 den bütük olan kayıtların class sütununu getirir
df.loc[df["age"] > 5, ["class", "age"]].head()
df.loc[(df["age"] > 5) & (df["sex"] == "male"), ["class",
                                                 "age"]].head()  # birden fazla koşul verilecekse hepsi parantez iiçne alınmalı ayrı ayrı
df.loc[(df["age"] > 5) & (df["sex"] == "male") & (df["who"] == "man"), ["class", "age", "who"]].head()
df.loc[(df["age"] > 2) & (df["sex"] == "male") & ((df["who"] == "man") | (df["who"] == "woman")), ["class", "age",
                                                                                                   "who"]].head()
##########GRUPLAMA
# count,first,last,mean,median,min,max,..
df["age"].mean()
df.groupby("sex")["age"].mean()  # cinsiyete göre gruplayıp ort alır
df.groupby("sex").agg({"age": "mean"})  # aynı işi yapar üsttekiyle !!
df.groupby("sex").agg({"age": ["mean", "sum"]})
df.groupby("sex").agg({"age": ["mean", "sum"], "survived": "mean"})
df.groupby(["sex", "embark_town"]).agg({"age": ["mean", "sum"], "survived": "mean"})
df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean", "sum"], "survived": "mean", "sex": "count"})

#######PİVOT TABLE

df.pivot_table("survived", "sex", "embarked")
df.pivot_table("survived", "sex", "embarked", aggfunc="std")
df.pivot_table("survived", "sex", ["embarked", "class"])

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 35, 40,
                                   90])  # kategorik değişkene çevrilmek istenen değişkeni sıralar ve gruplar
df.pivot_table("survived", "sex", ["new_age", "class"])
pd.set_option('display.width', 500)
df
###APPLY - LAMBDA :lambda kullan at fonks. apply satır sütuna fonk. uygular
df["age2"] = df["age"] * 6
df["age3"] = df["age"] * 7
df

(df["age"] / 10).head()

for col in df.columns:
    if "age" in col:
        df[col] = df[col] / 10  # print((df["age"] / 10).head())

df[["age", "age2", "age3"]].apply(lambda x: x ** 2)
df = df.drop("new_age", axis=1)
df.loc[:, df.columns.str.contains("age")].apply(
    lambda x: x / 2).head()  # bütün kolonları tarar age geçenlere fonk.u uygular
df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()  # standartlaştırma


def std_sclr(col_name):
    return (col_name - col_name.mean()) / col_name.std()


df.loc[:, df.columns.str.contains("age")].apply(std_sclr).head()  # dışarıda tanımlanan fonk. da kullanabilir
df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(std_sclr).head()
df
#########joın-merge
m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

df1
df2

pd.concat([df1, df2])

pd.concat([df1, df2], ignore_index=True)  # indexi sıfırlar

df1 = pd.DataFrame({"employess": ["ali", "veli", "seli"],
                    "grp": ["acc", "eng", "ceng"]})

df2 = pd.DataFrame({"employess": ["ali", "veli", "seli"],
                    "num": [1, 2, 3]})

df3 = pd.merge(df1, df2)
#######
data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data)
subset = df.loc[0:1, ['A', 'B']]
subset

#########
df = sns.load_dataset("titanic")
df.head()

# ilk başta uygulanabilcek genel fonksiyonlar;
##head, tail, shape,info,columns, ,ndex,describe,isnull
df.isnull().sum()


def check_df(dataframe, head=5):
    print(dataframe.shape)
    print(dataframe.dtypes)
    print(dataframe.head(head))
    print(dataframe.tail(head))
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)  # quantile değerleri
    print(dataframe.isnull().sum())


check_df(df)
df["embarked"].value_counts()  # kategoriler ve sayıları
df["sex"].unique()  # unique değerleri döndürür
df["class"].nunique()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "bool"]]
cat_cols

num_ut_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
num_ut_cat  # kategorik değişkenkleir bulduk

100 * df["survived"].value_counts() / len(df)  # oranlar

df["adult_male"].astype(int)  # true falseları 1 0 a dönüştürür

plt.show(block=True)  # grafik aktif

df[["age", "fare"]].describe().T
[col for col in df.columns if
 df[col].dtypes in ["int64", "float64"]]  # int ve float tipinde olan nümerik değişkenleri seçti

#############
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("titanic")
df.head()


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Parameters
    ----------
    dataframe
    cat_th
    car_th

    Returns
    -------

    """


##HEDEF DEĞ. ANALİZİ
num_cols = [col for col in df.columns if df[col].dtype in [int, float]]
corr = df[num_cols].corr()
sns.set(rc={'figure.figsize': (6, 6)})
sns.heatmap(corr, cmap="RdBu")
plt.show()
cor_matrix = df.corr().abs()
#######################
