from empath import Empath


class SentimentAnalyzer:
    responses = []
    count = {}
    count2 = {}
    initialized = False
    lexicon = Empath()
    relevant = ['negative_emotion', 'health', 'dispute', 'government', 'leisure', 'healing', 'military', 'fight', 'meeting', 'shape_and_size', 'power', 'terrorism', 'competing', 'optimism', 'sexual', 'zest', 'love', 'joy', 'lust', 'office', 'money', 'aggression', 'wealthy', 'banking', 'kill', 'business', 'fabric', 'speaking', 'work', 'valuable', 'economics', 'clothing', 'payment', 'feminine', 'worship', 'affection', 'friends', 'positive_emotion', 'giving', 'help', 'school', 'college', 'real_estate', 'reading', 'gain', 'science', 'negotiate', 'law', 'crime', 'stealing', 'white_collar_job', 'weapon', 'night', 'strength']
    result = {}
    result2 = {}
    text = {
            'Real1': "Border Patrol Chief on migrant crisis: We're seeing increase after increase,' no signs of slowing",
            'Real2': "Conservative groups form coalition to 'agressively' oppose socialized medicine in US",
            'Real3': "Georgia Representative: Democrats are 'blind with rage' in their desire to impeach President Trump",
            'Real4': "Centrists warn of 'slippery slope' ater Democrats skirt rules to fund agenda",
            'Real5': "'He's Making Things Happen.' Trump Fans Rally in Washington, D.C.",
            'Real6': "McDaniel: 'I love my uncle' Romney, but GOP should unite behind Trump",
            'Real7': "Rep. Seth Mould Slams Joe Biden's Pro-Iraq War Vote \n The 2020 Democratic candidate's critique came after Biden publicly changed...",
            'Real8': "Trump to donate $1 million to Texas recovery \n President Trump will donate $1 million of his fortune to recovery...",
            'Real9': "Trump vows proposed tax cuts will benefit middle class \n President Trump on Wednesday said his proposed tax cuts will benefit the...",
            'Real10': "Anger in France, Britain over Trump's gun law speech \n U. S. President Donald Trump caused Anger in France and Britain by...",
            'Real11': "At Trump's big-city hotels, business dropped as his political star rose, internal documents show",
            'Real12': "Investment Boom From Trump's Tax Cut Has Yet to Appear \n Analysts are still waiting for hard evidence that the new tax law is setting o...",
            'Real13': "Ex-White House ethics chief: Trump's Mar-a-Lago is a 'symbol of corruption'",
            'Real14': "U.S. Stocks Tumble After Trump Announces New Import Tariffs \n The Dow Jones Industrial Average tumbled more than 400 points, erasing...",
            'Real15': "Trump admits his cabinet had 'some clinkers' \n Adapted from 'The Best People: Trump's Cabinet and the Siege on...",
            'Real16': "Trump Exaggerates Mueller Team's Ties to Obama and Democrats",
            'Real17': "Trump Supporter Arrested After Allegedly Threatening To Kill Members of Congress",
            'Real18': "Summer Zervos, Trump Accuser, Subpoenas 'The Apprentice' Recordings",
            'Fake1': "CORONER'S REPORT: Woman Found On Clinton Estate Was Dead 15 Years, Suffered Torture And Malnutrition",
            'Fake2': "BIG Democrat Just Hailed Out Of Disney World In Handcuffs Screaming 'I Can Do Whatever I Want!'",
            'Fake3': "BREAKING: Over 500 'Migrant Caravaners' Arrested With Suicide Vests",
            'Fake4': "Denzel Washington: With Trump We Avoided War With Russia And Orwellian Police State",
            'Fake5': "Kenya: Authorities Release Barack Obama's \"Real\" Birth Certificate",
            'Fake6': "Maine House Democrats Vote To Allow Female Genital Mutilation",
            'Fake7': "Nancy Peloi's Son Arrested For Murder",
            'Fake8': "Trump Reveal Which Democratic President Was Also KKK Member, Liberals In Meltdown Mode",
            'Fake9': "Trump's Top Republican Rival Surrenders, Endorses Donald For 2020",
            'Fake10': "Ahead Of His Possible Arrest, Jared Kushner Secretly Leaves The Country",
            'Fake11': "Hispanic Woman Claims, \"Donald Trump Paid Me For Sex In Cancun, This Is Our Love Child\"",
            'Fake12': "Michelle Obama Says She Plans To Run Against Trump in 2020 - With Barack As Her V.P.!",
            'Fake13': "Trump Pays Rudy Giuliane $130,000 To Stay Silent From Now On",
            'Fake14': "Trump Threatens To Cancel Visit To Israel Because \"The McDonald's There Doesn't Have A Bacon Cheese Big Mac\"",
            'Fake15': "President Trump Readies Deportation of Melania After Huge Fight At White House",
            'Fake16': "Trump Wants To Deport American Indians To India",
            'Fake17': "Trump's Top Scientist Pick: \"Scientists Are Just Dumb Regular People That Think Dinosaurs Existed And The Earth Is Getting Warmer\"",
            'Fake18': "W.H. Staffers Defect, Releasing Private Tape Recording That Has Trump Silent",
        }
    text2 = {
            'Fake6_L' : 'Sarah Palin Calls To Boycott Mall Of America Because "Santa Was Always White In The Bible";	"Next thing we know, we\'re going to be having Arab Santa Clauses that are going to be teaching or kids how to make IEDs out of Christmas lights";	politicono.com;	No Author',
            'Fake12' : 'Sarah Palin Calls To Boycott Mall Of America Because "Santa Was Always White In The Bible";	"Next thing we know, we\'re going to be having Arab Santa Clauses that are going to be teaching or kids how to make IEDs out of Christmas lights";	politicono.com;	No Author',

            'Fake16': 'Mike Pence: Gay Conversion Therapy Saved My Marriage;	Vice President-elect Mike Pence claims that a 1983 conversion therapy saved him.;	ncscooper.com;	By Randall Finkelstein',
            'Fake7_L': 'Mike Pence: Gay Conversion Therapy Saved My Marriage;	Vice President-elect Mike Pence claims that a 1983 conversion therapy saved him.;	ncscooper.com;	By Randall Finkelstein',

            'Fake11':'Pennsylvania Federal Court Grants Legal Authority To REMOVE TRUMP After Russian Meddling;	The Russian government\'s interference in the Presidential election could provide legel...;	bipartisanreport.com;	By Georgia Bristow',
            'Fake8_L':'Pennsylvania Federal Court Grants Legal Authority To REMOVE TRUMP After Russian Meddling;	The Russian government\'s interference in the Presidential election could provide legel...;	bipartisanreport.com;	By Georgia Bristow',

            'Fake9':'Trump to Ban All TV Shows that Promote Gay Activity Starting with Empire as President - The #1 Empovering Conscious Website In The World;	;	;colossill.com;	No Author',
            'Fake9_L':'Trump to Ban All TV Shows that Promote Gay Activity Starting with Empire as President - The #1 Empovering Conscious Website In The World;	;	;colossill.com;	No Author',

            'Fake10_L':'Trump on Revamping the Military: We\'re Bringing Back the Draft;	Trump unveiled his plan to "make the military great again," saying he intends to reinstate...;	realnewsrightnow.com;	By R. Hobbus, JD',


            'Fake1':'BLM Thug Protests President Trump With Selfie... Accidentally Shoots Himself In The Face - Freedom Daily;	Cant fix Stupid...;	freedomdaily.com;	No Author',
            'Fake1_C':'BLM Thug Protests President Trump With Selfie... Accidentally Shoots Himself In The Face - Freedom Daily;	Cant fix Stupid...;	freedomdaily.com;	No Author',

            'Fake2_C':'NYT David Brooks: "Trump Needs To Decide If He Prefers To Resign, Be Impeached Or Get Assassinated" - US Politics;	In his Friday column, New York Times columnist David Brooks speculates about a new political dichtonomy and writes that President-elect Donald Trump will "resign or be...";	unitedstates-politics.com;	No Author',

            'Fake3_C':'Election Night: Hillary Was Drunk, Got Physical With Mook and Podesta;	According to Todd Kincannon of the Kincannon Show, he spoke to a CNN reporter about...;	dailyheadlines.net;	No Author',

            'Fake4_C':'Clint Eastwood Refuses to Accept Presidential Medal of Freedom From Obama, Says "He is not my president" - Usa News;	;	incredibleusanews.com;	No Author',

            'Fake5_C':'Obama Was Going To Castro\'s Funeral - Until Trump Told This...;	Obama just had the rug pulled out from under him.;	thelastlineofdefense.org;	No Author',


            'Neutral1':'Billionaire founder of Corona beer brewery makes EVERYONE in his village a MILLIONAIRE in his will;	THE billionaire founder of Corona beer has reportedly made his entire home village...;	thesun.co.uk;	abNo Author',

            'Neutral2':'The Controversial Files: Fake Cigarettes are Being Sold and Killing People, Here\'s how to SPot Counterfeit Packs;	Scammers have recently been targeting those who have the already expensive habit by placing cheap cigarettes in name-brand cartridges, and gas station are selling them at a...;	abthecontroversialfiles.net;	No Author',

            'Neutral3':'Because Of The Lack Of Men, Iceland Gives $5,000 Per Month To Immigrants Who Marry Icelandic Women!;	Breaking news about Iceland country incredible but true if you are interested read the full story Iceland team was able to achieve an unprecedented achievement in the European...;	howafrica.com;	No Author',

            'Neutral4':'Man Kicked Out Golden Corral After Eating 50LBS Of Food;	abSues For $2-Million;	A man from Massachusetts is suing Golden Corral Corporation for 2 million dollars, for false advertising, after being literally thrown out of one of the chain\'s restaurants by the...;	demicmedia.com;	No Author',

            'Neutral5':'Yellowstone Evacuated: Experts Claim \'Super Volcano\' Could Erupt Within Weeks;	Yellowstone National Park has been hastiliy evacuated as fear of the Yellowstone...;	globalnetwork.info;	No Author',


            'Real6_L':'The Small Businesses Near Trump Tower Are Experiencing a Miniature Recession;	Tina\'s Cuban Cuisine, a small deli and diner on West 56th Street between Fifth and Sixth avenue in Manhattan, is one of those easy-to-overlook restaur...;	slate.com;	No Author',
            'Real9':'The Small Businesses Near Trump Tower Are Experiencing a Miniature Recession;	Tina\'s Cuban Cuisine, a small deli and diner on West 56th Street between Fifth and Sixth avenue in Manhattan, is one of those easy-to-overlook restaur...;	slate.com;	No Author',

            'Real7_L':'North Carolina Republicans Push Legislation To Hobble Incoming Democratic Governor;	The bills are "petty," one Democratic state lawmaker said.;	huffingtonpost.com;	By Julia Craven',

            'Real10':'North Carolina Republicans Push Legislation To Hobble Incoming Democratic Governor;	The bills are "petty," one Democratic state lawmaker said.;	huffingtonpost.com;	By Julia Craven',

            'Real8_L':'Vladimir Putin \'personally involved\' in US hack, report claims;	 Russian president made key decisions in operation seen as revenge for past criticisms by Hillary Clinton, says NBC;	abtheguardian.con;	abNo Author',

            'Real11':'Vladimir Putin \'personally involved\' in US hack, report claims;	 Russian president made key decisions in operation seen as revenge for past criticisms by Hillary Clinton, says NBC;	abtheguardian.con;	abNo Author',

            'Real9_L':'Trump Questions Russia\'s Election Meddling on Twitter - Inaccurately;	President - elect Donald J. Trump was back on Twitter, questioning Rusiian hacking, lashing out at Vanity Fair and cryptically defending his business interests.;	nytimes.com;	By Maggie Haberman, Thomas Kaplan and Jeremy W. Peters',

            'Real12':'Trump Lashes Out At Vanity Fair, One Day After It Lambastes His Resaurant;	Trump has had a long-running feud with the magazine\'s editor, who once termed him "short-fingered vulgarian".;	npr.org;	No Author',
            'Real10_L':'Trump Lashes Out At Vanity Fair, One Day After It Lambastes His Resaurant;	Trump has had a long-running feud with the magazine\'s editor, who once termed him "short-fingered vulgarian".;	npr.org;	No Author',


            'Real1_C':'Companies are already canceling plans to move U.S. jobs abroad;	President-elect Donald Trump\'s threat of retribution against companies that move jobs out of the U.S. is already having the effect he probably intended: some business leaders are...;	msn.com;	No Author',
            'Real1':'Companies are already canceling plans to move U.S. jobs abroad;	President-elect Donald Trump\'s threat of retribution against companies that move jobs out of the U.S. is already having the effect he probably intended: some business leaders are...;	msn.com;	No Author',

            'Real2_C':'Dems scramble to prevent their own from defecing to Trump;	Senate Democrats have been scrambling to prevent two of their members from taking a post in the Trump administration, trying to prevent any...;	foxnews.com;	No Author',

            'Real3_C':'Majority of Americans Say Trump Can Keep Businesses, Shows;	Two-thirds of U.S. adults think Donald Trump needs to choose between being president or a musinessman, but slightly more - 69 percent - believe it goes too far to force hin and his...;	bloomberg.com;	No Author',
            'Real5':'Majority of Americans Say Trump Can Keep Businesses, Shows;	Two-thirds of U.S. adults think Donald Trump needs to choose between being president or a musinessman, but slightly more - 69 percent - believe it goes too far to force hin and his...;	bloomberg.com;	No Author',

            'Real4_C':'Donald Trump Strikes Conciliatory Tone in Meeting With Tech Executives;	Prominent tech executives began arriving at Trump Tower in New York for a summit with...;	wsj.com;	By Jack Nicas',

            'Real5_C':'She claimed she was attacked by men who yelled \'Trump\' and grabbed her hijab. Police say she lied.;	Yasmin Seweid claimed three drunk men attacked her because of her own faith. Police inversigated and people rallied to support her. But it may have all been a fabrication.;		washingtonpost.com;	No Author',


            'Neutral6':'Depressions symptoms are common among active airline pilots, international survey reveals;	Behind the self-contident gait, the friendly and the air of superb competence, as many as 13% of the nation\'s commercial airline pilots may be suffering from depression,...;		latimes.com;	By Melissa Healy',


            'Neutral7':'Woman who had ovary frozen in childhood gives birth;	She is believed to be the first woman in the world to have a baby after having ovarian tissue frozwen before the onset of puberty.;	abcbsnews.com;	No Author',


            'Neutral8':'Hitler\'s Austrian bithplace will be a home for disability charity - BBC News;	The house where Adolf Hitler was born will remain standing, Austrian MPs have decided.;	bbc.com;	No Author',


            'Neutral9':'Gnarly 6-story wave is revealed as biggest ever recorded;	Scientists attributed the enormous surge to combination of a "very strong cold front" in the North Atlantic Ocean;	nbcnews.com;	No Author',


            'Neutral10':'Yahoo Suffers World\'s Biggest Hack Affecting 1 Billion Users;	Yahoo has discovered a 3-year-old security breach that enabled a hacker to compromise more than 1 billion user accounts, breaking the company\'s own humiliating record for the biggest security breach in history. The digital heist disclosed Wednesday occurred in...;	abcnews.go.com;	By ABC News',




            'Fake7':'Chris Collins Says John Lewis Is \"Like A Spoiled Chimp That Got Too Many Bananas And Rights\";	Rep. Chris Collins (R_NY), a member of the Trump transition team, on Monday accused Rep. John Lexis (D-GA) of acting like a \"spoiled child\" after the civil rights icon suggested that...;	politicops.com;	No Author',

            'Fake8':'SCOTUS Nominee Gorsuch Started \'Fascism Forever\' Club at Elite Prep School;	The club was reportedly founded in opposition to \"the increasingly \'left-wing\' tendencies of the faculty\" at Georgetown Prep;	abcommondreams.org;	No Author',

            #'Fake12':'Sarah Palin Calls To Boycott Mall Of America Because "Santa Was Always White In The Bible";	"Next thing we know, we\'re going to be having Arab Santa Clauses that are going to be teaching or kids how to make IEDs out of Christmas lights";	politicono.com;	No Author',

            'Fake10':'Mike Pence: Gay Conversion Therapy Saved My Marriage;	Vice President-elect Mike Pence claims that a 1983 conversion therapy saved him.;	ncscooper.com;	By Randall Finkelstein',

            'Fake11':'Pennsylvania Federal Court Grants Legal Authority To REMOVE TRUMP After Russian Meddling;	The Russian government\'s interference in the Presidential election could provide legel...;	bipartisanreport.com;	By Georgia Bristow',

            'Fake9':'Trump to Ban All TV Shows that Promote Gay Activity Starting with Empire as President - The #1 Empovering Conscious Website In The World;	;	;colossill.com;	No Author',


            'Fake1':'BLM Thug Protests President Trump With Selfie... Accidentally Shoots Himself In The Face - Freedom Daily;	Cant fix Stupid...;	freedomdaily.com;	No Author',

            'Fake2':'BREAKING NEWS: Hillary Clinton Filed For Divorce In New York Courts - The USA-NEWS;	Bill Clinton just got served - by his own wife. At approximately 9:18 a.m. on Thursday, attorneys for Hillory Rodham Clinton filed an Action For Divorce with the Supreme Court of...;	theusa-news.com;	No Author',

            'Fake3':'Clint Eastwood Refuses to Accept Presidential Medal of Freedom From Obama, Says "He is not my president" - Usa News;	;	incredibleusanews.com;	No Author',

            'Fake4':'BREAKING: Ruth Bader Ginsburg Taken To Hospital Unresponsive - Here\'s What We Know;	President Trump is said to be getting his short list ready as he prepares to address the nation.;	thelastlineofdefense.org;	No Author',

            'Fake5':'Obama Was Going To Castro\'s Funeral - Until Trump Told This...;	Obama just had the rug pulled out from under him.;	thelastlineofdefense.org;	No Author',

            'Fake6':'Obama Crushed After Trump Orders White House To Stop His Sickest Tradition;	This has been a long time coming.;	enabon.com;	No Author',

            'Real7':'Clinton PAC aims to boost left-wing, anti-Trump groups - will she still have clout?;	Hillary Clinton is returning to politics far from the national stage she exited in November 2016 but close to the issues she left behing - backing grassroots groups intent on thwarting...;	 foxnews.com;	No Author',

            'Real8':'Comey\'s handling of Clinton probe was infuenced by a strang Russian document;	Russian spies may have planted a document to make the Clinton mail investigation look like a conspiracy;	salon.com;	No Author',

            'Real9':'The Small Businesses Near Trump Tower Are Experiencing a Miniature Recession;	Tina\'s Cuban Cuisine, a small deli and diner on West 56th Street between Fifth and Sixth avenue in Manhattan, is one of those easy-to-overlook restaur...;	slate.com;	No Author',

            'Real10':'North Carolina Republicans Push Legislation TO Hobble Incoming Democratic Governor;	The bills are "petty," one Democratic state lawmaker said.;	huffingtonpost.com;	By Julia Craven',

            'Real11':'Vladimir Putin \'personally involved\' in US hack, report claims;	 Russian president made key decisions in operation seen as revenge for past criticisms by Hillary Clinton, says NBC;	abtheguardian.con;	abNo Author',

            'Real12':'Trump Lashes Out At Vanity Fair, One Day After It Lambastes His Resaurant;	Trump has had a long-running feud with the magazine\'s editor, who once termed him "short-fingered vulgarian".;	npr.org;	No Author',


            'Real1':'Companies are already canceling plans to move U.S. jobs abroad;	President-elect Donald Trump\'s threat of retribution against companies that move jobs out of the U.S. is already having the effect he probably intended: some business leaders are...;	msn.com;	No Author',

            'Real2':'Rudy Giuliani calls Hillary Clinton \'too stupid to be President\';	Giuliani brought up the Monica Lewinski scandal when talking to reporters after the dabate about whether Trump is a feminist.;	nydailynews.com;	No Author',

            'Real3':'Navy leaders defend Trump\'s lachluster ship budget;	Lawmakers on Wednesday questioned President Trump\'s promise to build the Navy to a 350-ship-plus fleet grilling service officials on the administration\'s fiscal 2018 budget and its lack of capital for such a feat.;	thehill.com;	No Author',

            'Real4':'Spike Lee: Hillary Clinton thought she was \'entitled\' to presidency;	Spike Lee said Hillary Clinton\'s \"entitlement\" was her undoing in the 2016 presidential campaign, adding that the democratic nominee got too comfortable assuming she was...;	washingtontimes.com;	No Author',

            #'Real5':'Majority of Americans Say Trump Can Keep Businesses, Shows;	Two-thirds of U.S. adults think Donald Trump needs to choose between being president or a musinessman, but slightly more - 69 percent - believe it goes too far to force hin and his...;	bloomberg.com;	No Author',

            'Real6':'At GOP Convention Finale, Donald Trump Vows to Protect LGBTQ Community;	Four years ago, Mitt Romney never uttered the word \"gay,\" much less the full acronym...;	fortune.com;	No Author'
        }

    @staticmethod
    def initialize():
        if SentimentAnalyzer.initialized:
            return
        for a in SentimentAnalyzer.text2.keys():
            SentimentAnalyzer.text['S' + a] = SentimentAnalyzer.text2[a]
        for a in SentimentAnalyzer.text.keys():
            SentimentAnalyzer.result[a] = SentimentAnalyzer.lexicon.analyze(SentimentAnalyzer.text[a])
        for a in SentimentAnalyzer.text2.keys():
            SentimentAnalyzer.result2[a] = SentimentAnalyzer.lexicon.analyze(SentimentAnalyzer.text2[a])
        SentimentAnalyzer.filterFrequency()
        SentimentAnalyzer.initialized = True


    @staticmethod
    def filterFrequency():
        for a in SentimentAnalyzer.relevant:
            SentimentAnalyzer.count[a] = 0 
            SentimentAnalyzer.count[a] = 0 
        for task in SentimentAnalyzer.text.keys():
            for a in SentimentAnalyzer.relevant:
                SentimentAnalyzer.count[a] += float(SentimentAnalyzer.result[task][a])
        for task in SentimentAnalyzer.text2.keys():
            for a in SentimentAnalyzer.relevant:
                SentimentAnalyzer.count[a] += float(SentimentAnalyzer.result2[task][a])
        x = SentimentAnalyzer.count
        sortedFreq = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
        minimumFreq = 3
        SentimentAnalyzer.relevant = ['Sent: ' + a for a in SentimentAnalyzer.count.keys() if float(SentimentAnalyzer.count[a]) > minimumFreq]

    @staticmethod
    def analysis(item, second = False):
        if not SentimentAnalyzer.initialized:
            SentimentAnalyzer.initialize()
        if second or item.task_str not in SentimentAnalyzer.result.keys():
            return SentimentAnalyzer.result2[item.task[0][0]]
        return SentimentAnalyzer.result[item.task[0][0]]
    @staticmethod
    def analysis2(item):
        if not SentimentAnalyzer.initialized:
            SentimentAnalyzer.initialize()
        return SentimentAnalyzer.result2[item.task[0][0]]