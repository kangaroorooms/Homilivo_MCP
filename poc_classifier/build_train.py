"""Writes data/train.jsonl (fresh handwritten examples). ctx mode per label:
T = has_room_context True, F = False, M = alternating (mixed)."""
import json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
D = {}
D["NEW_SEARCH"] = ("M", """
i am looking for a flat in pune
need pg in hyderabad near hitech city
anyone know a good place to rent in chennai
room for rent in noida sector 62
looking for 1bhk under 12k in gurgaon
can you find me a studio apartment in mumbai
need accommodation near my college in jaipur
single room in delhi for 8000
want to rent a 3bhk in kolkata
mujhe pune mein flat chahiye
mumbai me pg dhundh raha hu
delhi mein sasta kamra batao
hyderabad mein 2bhk rent pe chahiye
mujhe ek room chahiye indore me
koi pg hai kya andheri ke paas
plz fnd me a rom in chenai
lookin for flatt in pne under 15k
need rum in hydrabad asap
rooms avilable in bangalore??
i i need a a flat in delhi
um so like i want a room in mumbai you know
flat flat in pune 2 bhk
looking for room... in noida... cheap
need place stay lucknow budget 6000
i want to move to bangalore next month need a pg
show me rooms in whitefield
find pg for girls in koramangala
room chahiye jaipur me student ke liye
suggest some flats in thane under 25000
accommodation in goa for 3 months
hey i wanna rent a room in kochi
fresh search: 1rk in ahmedabad
actually forget those, find me flats in surat
कमरा चाहिए पुणे में
नोएडा में पीजी बताओ
""")
D["REFINEMENT"] = ("T", """
make it under 9000
only 2bhk ones please
i want it furnished
budget max 18k
restrict to girls pg
only near the metro
within 5 km of the airport
below 12000 rupees
add the filter for parking
ac rooms only
increase the budget to 25k
narrow it to single rooms
bas 1bhk dikhao
budget 10 hazaar tak
furnished wale hi chahiye
sirf ladkiyon ke liye wale
metro ke paas wale
15k se kam wale dikhao
under 8k only
cheaper ones plz
only those with balcony
less than 20000 and fully furnished
change it to 3bhk
umm only the furnished ones
udr 11k only
can you filter by pet friendly
ground floor wale hi
anything near indiranagar instead
only unfurnished
do the same search but for 2bhk
रेंट 12000 से कम वाले
सिर्फ फर्निश्ड वाले
""")
D["SHOW_MORE_OPTIONS"] = ("T", """
show me some more
any other rooms
got more listings
let me see other options
more please
next ones
show remaining rooms
do you have others
anything more to show
i want to see further options
aur dikhao
aur options hai kya
kuch aur batao
or koi room hai
or dikhao na
agle wale dikhao
baaki ke rooms bhi dikhao
more rooms plz
sho me mor
ane other optons
what other places are there
can you list more
another set of rooms please
any alternatives
keep going show others
um more more
और दिखाओ
और कमरे हैं क्या
other listings pls
dikhao aur kuch
""")
D["REJECT_CURRENT"] = ("T", """
i dont like any of these
these are not what i wanted
nothing here works for me
not interested in these rooms
all of them are bad
none suits me
these are useless
no i dont want any of them
not happy with these options
skip all of these
inme se koi nahi chahiye
ye sab pasand nahi aaye
koi bhi theek nahi hai
ye rooms mujhe nahi chahiye
sab bekaar hai
mujhe inme se kuch nahi jamta
nahi nahi ye nahi
non of them good
non of thes r fine
dont like them
no no not these
these dont match what i need
nope none of it
disappointing options sorry not for me
ye koi kaam ke nahi
कोई भी पसंद नहीं आया
ये सब नहीं चाहिए
reject these all
not these not these
nothing here is right
""")
D["LOWEST_PRICE"] = ("T", """
which one is the cheapest
lowest rent among these
the least expensive room please
tell me the most affordable one
which has minimum rent
cheapest of all of them
give me the lowest priced option
which room costs the least
sabse sasta kaunsa hai
sabse kam rent kiska hai
inme sabse sasta batao
kam price wala kaunsa
cheepest one
whch is the cheapst
cheapest cheapest which
cheapest one pls fast
the budget friendliest of these
minimum price wala
सबसे सस्ता कौन सा है
सबसे कम किराया
which is least costly
lowest priced one of them
pick the cheapest for me
sabse sasta wala dikhao
tell me which is cheapest um one
the one with lowest price
lowest rental please
sasta wala kon sa h
which of these is most economical
cheapest room from the list
""")
D["HIGHEST_PRICE"] = ("T", """
which one is the most expensive
highest rent among these
the costliest room please
tell me the priciest one
which has maximum rent
most expensive of all of them
give me the highest priced option
which room costs the most
sabse mehnga kaunsa hai
sabse zyada rent kiska hai
inme sabse mehnga batao
zyada price wala kaunsa
priciest one
whch is the most expnsive
costliest costliest which
highest price one pls
the most premium of these
maximum price wala
सबसे महंगा कौन सा है
सबसे ज्यादा किराया
which is most costly
highest priced one of them
pick the most expensive for me
sabse mehnga wala dikhao
tell me which is priciest um one
the one with highest price
highest rental please
mehnga wala kon sa h
which of these is the most luxurious in price
most expensive room from the list
""")
D["ROOM_PRICE"] = ("T", """
how much is the rent
what is the price of this room
what does the first one cost
rent for the second room
how much per month is it
price of the pg please
tell me the monthly rent
what's the deposit and rent
how much do they charge for this one
cost of the 2bhk
iska kiraya kitna hai
ye kitne ka hai
is room ka rent batao
first wale ka price kya hai
mahine ka kitna padega
dusre room ka kiraya
price btao
hw much is rnt
wat is the pricing
rent kya hai iska
how much for third one
monthly charges for this
what will i pay each month
isme kitna lagega
इसका किराया कितना है
किराया बताओ
cost per month of that room um
the price pls
how much does it cost to stay there
rent amount of the pg
""")
D["ROOM_TYPE"] = ("T", """
what type of room is it
is this a 1bhk or 2bhk
is it a pg or a flat
tell me the room type
what kind of accommodation is the second one
is that a single room or shared
how many bedrooms does it have
bhk of this one
is it a studio
kya ye 2bhk hai
ye pg hai ya flat
room ka type kya hai
kitne bedroom hai isme
single hai ya sharing
kis type ka room hai ye
wat type of room
is it sharing or private
is this 1rk
type btao
which bhk is the third
number of bedrooms in the first one
is it shared with others
kya ye ek kamre ka hai
ye flat hai ya pg
is this a studio apartment or normal
room type of that one pls um
bhk kitne hai
बेडरूम कितने हैं
ये कौन सा टाइप है
sharing wala hai kya
""")
D["LOCATION"] = ("T", """
where is this room located
what is the address
which area is the second one in
how far is it from the metro
location of the first one
is it near the railway station
which locality is this
is any of these close to the airport
tell me the area of the pg
kaha pe hai ye room
iska address kya hai
ye kis area me hai
metro se kitna door hai
station ke paas hai kya
location batao
wher is it
how far from my office in whitefield
is this near a market
what neighbourhood is it
which room is closest to the college
nearest one to the bus stand
kaun sa room mall ke paas hai
area ka naam batao
ye jagah kaha hai
is it in a safe locality
यह कहाँ है
पता बताओ
locaton of the thrid
address pls um
how far is the first one from the beach
""")
D["AMENITY"] = ("T", """
does it have air conditioning
is there parking
do they provide meals
any gym in the building
is laundry included
does the room have attached bathroom
is there a lift
what facilities are available
is power backup there
does this one allow cooking
kya isme ac hai
parking milti hai kya
khana milta hai isme
geyser hai kya
isme kya kya facilities hai
lift hai ya nahi
wifi included hai kya
is thre wifi
do it have fridge
has it got a washing machine
tv available
does the second room have a balcony
water supply 24 hours
cctv or security guard
kya furniture diya hai
क्या इसमें एसी है
पार्किंग है क्या
is it furnished with bed and wardrobe
amenities list pls
any housekeeping service um
""")
D["STUDENT_SUITABILITY"] = ("T", """
is this good for students
which of these is best for a college student
can students stay here
suitable for a student
any room here that suits students
is it student friendly
which one would a student prefer
can i stay here as a student
is it ok for college going boys
ye students ke liye theek hai kya
student ke liye kaunsa best hai
kya ye college students ke liye hai
inme se kaunsa student friendly hai
students allowed hai kya
college wale rah sakte hai kya
is it fine for stdents
good for studnt
student fit room which
suit a freshers batch
does this suit a student budget
is it near enough for students to walk to college
a student like me can live here
ye wala student ke layak hai kya
which is ideal for students
छात्रों के लिए ठीक है क्या
स्टूडेंट्स के लिए कौन सा सही है
is the second one meant for students
students okay um in this
hostel type for students
any of these for college kids
""")
D["FAMILY_SUITABILITY"] = ("T", """
is this good for a family
which of these is best for a family of four
can a family stay here
suitable for me my wife and kids
any room here that suits families
is it family friendly
which one would a family prefer
can i stay here with my parents
is it ok for a married couple
ye family ke liye theek hai kya
family ke liye kaunsa best hai
kya ye parivar ke liye hai
inme se kaunsa family friendly hai
families allowed hai kya
bacchon ke saath rah sakte hai kya
is it fine for famly
good for famly stay
family fit room which
suit a couple with a baby
does this suit a family budget
is it safe for kids
a family like ours can live here
ye wala family ke layak hai kya
which is ideal for families
परिवार के लिए ठीक है क्या
फैमिली के लिए कौन सा सही है
is the second one meant for families
family okay um in this
spacious enough for four people
any of these for joint family
""")
D["PROFESSIONAL_SUITABILITY"] = ("T", """
is this good for working professionals
which of these is best for an office goer
can professionals stay here
suitable for a working person
any room here that suits IT employees
is it professional friendly
which one would a working bachelor prefer
can i stay here as a working man
is it ok for job holders
ye job wale logo ke liye theek hai kya
office jane walo ke liye kaunsa best hai
kya ye working professionals ke liye hai
inme se kaunsa professional friendly hai
working log allowed hai kya
naukri wale rah sakte hai kya
is it fine for profesionals
good for workin people
professional fit room which
suit a software engineer
does this suit an employee budget
is it quiet enough for work from home
a professional like me can live here
ye wala working ke layak hai kya
which is ideal for professionals
नौकरी करने वालों के लिए ठीक है क्या
वर्किंग प्रोफेशनल्स के लिए कौन सा सही है
is the second one meant for professionals
professionals okay um in this
good for someone who works night shifts
any of these for corporate people
""")
D["ROOM_DETAILS"] = ("T", """
tell me more about the first one
give me full details of this room
describe the second room
i want to know about the third listing
more info on that pg
what's special about the first room
details please
explain the 2bhk option
is the first one available right now
can you elaborate on this room
pehle wale ke bare me batao
is room ki puri jankari do
dusre wale ka detail do
teesra wala kaisa hai
ye wala dikhao thoda detail me
isme kya khaas hai
i like the first one tell me more
lets go with the second one
i pick the third room
the last one how is it
tell me abt the 1st
detials of room 2
is it vacant
availability of the second
ye available hai kya abhi
पहले वाले के बारे में बताओ
इसकी पूरी जानकारी दो
describe it um properly
give me everything on this property
the one in koramangala tell me about it
which one is ready to move in
""")
D["ROOM_TABLE"] = ("T", """
put them in a table
show a comparison table
list everything in rows and columns
can i get a tabular view
table format please
give me a summary table of these rooms
display the rooms as a grid
overview of all rooms side by side in a table
make a chart of the rooms with rent and area
tabular summary pls
sabko table me dikhao
table banao inka
table format me batao
sab rooms ki list table me do
ek table me sab dikhao
tabel of all rooms
show in tabel
rooms in a sheet format
list all of them with price and location in a table
can you tabulate them
give me a table pls um
सब कुछ टेबल में दिखाओ
टेबल बनाओ
all rooms with details in table form
spreadsheet style view of these
one table with all the options
table of the three rooms
table with rent type and area
please show it as a table not text
quick table of everything
""")
D["COMPARE_ROOMS"] = ("T", """
compare room 1 and room 2
how does the first differ from the second
which is better the pg or the 2bhk
compare all three rooms
difference between the first two
first vs second which one wins
compare them on price and location
what is the difference between these rooms
is the second better than the third
pros and cons of each room
pehle aur dusre ki tulna karo
inme kya fark hai
dono me se kaunsa behtar hai
first aur third ko compare karo
sabka comparison do
compar the 2 rooms
compare thes plz
which one is more worth it
rank these rooms for me
help me choose between the first and second
which of the two should i pick
um compare them both
side by side comparison please
how are these rooms different
क्या फर्क है इन दोनों में
दोनों की तुलना करो
2bhk versus pg compare
compare koramangala one with hsr one
better value for money which of these
weigh the first against the last
""")
D["ACKNOWLEDGMENT"] = ("M", """
ok
okay got it
alright
cool
thank you
thanks a lot
fine
understood
i see
sure
great
nice
perfect
good to know
noted
okay thanks
achha
theek hai
ok ji
accha samajh gaya
thik hai
shukriya
dhanyavad
badhiya
bohot badiya
thx
tnx
k
okayyy
ओके
ठीक है
धन्यवाद
alright cool thanks
ohh i see okay
""")
D["GREETING"] = ("M", """
hello
hey
hey there
good morning
good evening
good afternoon
hello there
hiya
yo
hi bot
hello assistant
greetings
namaste
namaskar
hello ji
hii
heyy
helo
hellooo
hi hi
hey whats up
howdy
sup
good day
salaam
नमस्ते
हेलो
हाय
suprabhat
hello hello anyone there
hey hi
hi there friend
""")
D["CLOSING"] = ("M", """
bye
goodbye
see you later
that's all for now
i'm done
talk to you later
bye bye
thats it thanks bye
i am leaving now
end chat
exit
quit
close the chat
we are done here
see ya
take care bye
good night
chalo bye
alvida
ok bye
phir milte hai
chalta hu bye
bas itna hi tha bye
chat band karo
ab main chalta hu
tata
byee
cya
gtg bye
ठीक है अलविदा
बाय
bas ho gaya dhanyavad bye
""")
D["PLATFORM_INFO"] = ("M", """
what is homilivo
who are you
what can you do
how does this platform work
how do i list my property
is there a fee for tenants
what is the refund policy
how do you verify listings
are the owners verified
is my data safe with you
what cities do you cover
how can i contact support
do you charge brokerage
is homilivo free to use
who built this chatbot
tell me about your service
how are listings approved
what is the cancellation policy
homilivo kya hai
tum kaun ho
tum kya kar sakte ho
yaha brokerage lagti hai kya
listing kaise verify hoti hai
kya ye app free hai
apni property kaise add karu
what is homilivo exactly um
wat does homlivo do
how do payments work on this site
what are your terms and conditions
होमिलिवो क्या है
आप क्या कर सकते हैं
""")
D["UNSUPPORTED_ACTION"] = ("M", """
book this room for me
reserve the second one
i want to pay the deposit now
schedule a visit for tomorrow
call the owner for me
send my phone number to the landlord
cancel my booking
confirm my reservation
arrange a viewing this weekend
message the owner that i am interested
sign the rental agreement for me
transfer the advance payment
book a site visit at 5pm
email the broker my details
mera booking cancel kar do
is room ko book karo
owner ko call lagao
advance payment kar do
visit schedule karo kal ke liye
agreement sign kardo
owner ko message bhejo
pay rent online for me
boook it now
resrve this room
shedule a visit
get me a callback from the owner
add this room to my wishlist and book it
hold this room for me till friday
रूम बुक कर दो
मालिक को फोन करो
""")
D["OUT_OF_SCOPE"] = ("M", """
what is the weather today
tell me a joke
who won the cricket match yesterday
write me a python function to sort a list
what is the capital of france
solve 45 times 12
recommend a good movie
how do i cook biryani
translate good morning into french
what is the stock price of tcs
tell me about the history of rome
give me a poem about the sea
who is the prime minister of india
how to lose weight fast
best phone under 20000
aaj ka mausam kaisa hai
ek joke sunao
cricket ka score batao
biryani kaise banate hai
mujhe gaana sunao
what is wether tomorow
tel me a jok
explain quantum physics
how to learn guitar
bitcoin price now
what time is it in tokyo
help me write an email to my boss
flight tickets to dubai
आज का मौसम कैसा है
कोई अच्छी फिल्म बताओ
""")
D["UNCLEAR"] = ("M", """
hmm
umm
uh
idk
what
huh
asdf
qwerty
blah blah
...
?
!!
so
the
hmmmm maybe
jkjkjk
aaaaa
xyz
mm
ok but
and then
um uh well
dunno
pata nahi
kuch bhi
hmm kya bolu
kjhgf
lol
zzz
yeah no
well
या
ह्म्म
as per the
,,,
""")

rows = []
for lab, (mode, items) in D.items():
    items = [s.strip() for s in items.strip().splitlines() if s.strip()]
    assert len(items) >= 25, (lab, len(items))
    for i, t in enumerate(items):
        ctx = {"T": True, "F": False, "M": i % 2 == 0}[mode]
        # context-dependent labels: some appear without context (user asked too early)
        if mode == "T" and i % 7 == 3:
            ctx = False
        rows.append({"text": t, "label": lab, "has_room_context": ctx})

with open(os.path.join(HERE, "data", "train.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), sorted(Counter(r["label"] for r in rows).items()))
