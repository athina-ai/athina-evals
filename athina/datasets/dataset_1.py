data = [
    {
        "query": "Which is better the Free Software Movement or the Open Source Initiative?",
        "context": "\"Both the modern free software movement and the Open Source Initiative were born from a common history of Unix, Internet free software, and the hacker culture, but their basic goals and philosophy differ, the free software movement being more focused on the ethics of software, and their open source counterparts being more focused on practical benefits. The Open Source Initiative chose the term \\\"open source,\\\" in founding member Michael Tiemann's words, to \\\"dump the moralizing and confrontational attitude that had been associated with 'free software'\\\" and instead promote open source ideas on \\\"pragmatic, business-case grounds.\\\"\"",
        "response": "The Free Software Movement is the better organization for me since I also hold a high value on ethics and believe in that vision.",
        "prompt_slug": "closed_qa"
    },
    {
        "query": "Given this paragraph describing the conclusion of the movie Groundhog Day, how did Phil manage to break the cycle and wake up on February 3 instead of back once again on February 2?",
        "context": "\"During one iteration of the loop, Phil reports on the Groundhog Day festivities with such eloquence that other news crews stop working to listen to his speech, amazing Rita. Phil continues his day helping the people of Punxsutawney. That night, Rita witnesses Phil's expert piano-playing skills as the adoring townsfolk regale her with stories of his good deeds. Impressed by his apparent overnight transformation, Rita successfully bids for him at a charity bachelor auction. Phil carves an ice sculpture in Rita's image and tells her that no matter what happens, even if he is trapped in the loop forever, he is finally happy because he loves her. They share a kiss and retire to Phil's room. He wakes the next morning to \\\"I Got You Babe\\\" but finds Rita is still in bed with him and the radio banter has changed; it is now February 3. Phil tells Rita that he wants to live in Punxsutawney with her.\\nhttps://en.wikipedia.org/wiki/Groundhog_Day_(film)\"",
        "response": "This paragraph summarizing the conclusion of Groundhog Day explains what was happening when Phil wakes up on February 3 but does not explain why he is able to escape the loop.",
        "prompt_slug": "closed_qa"
    },
    {
        "query": "Given this reference text about the Koala, tell me what makes up most of their diet, how long joeys spend in their mothers' pouches and what are some of the pathogens they are threatened by.",
        "context": "\"The koala or, inaccurately, koala bear (Phascolarctos cinereus), is an arboreal herbivorous marsupial native to Australia. It is the only extant representative of the family Phascolarctidae and its closest living relatives are the wombats. The koala is found in coastal areas of the mainland's eastern and southern regions, inhabiting Queensland, New South Wales, Victoria, and South Australia. It is easily recognisable by its stout, tailless body and large head with round, fluffy ears and large, spoon-shaped nose. The koala has a body length of 60\\u201385 cm (24\\u201333 in) and weighs 4\\u201315 kg (9\\u201333 lb). Fur colour ranges from silver grey to chocolate brown. Koalas from the northern populations are typically smaller and lighter in colour than their counterparts further south. These populations possibly are separate subspecies, but this is disputed.\\n\\nKoalas typically inhabit open Eucalyptus woodland, as the leaves of these trees make up most of their diet. This eucalypt diet has low nutritional and caloric content and contains toxins that deter most other mammals from feeding on it. Koalas are largely sedentary and sleep up to twenty hours a day. They are asocial animals, and bonding exists only between mothers and dependent offspring. Adult males communicate with loud bellows that intimidate rivals and attract mates. Males mark their presence with secretions from scent glands located on their chests. Being marsupials, koalas give birth to underdeveloped young that crawl into their mothers' pouches, where they stay for the first six to seven months of their lives. These young koalas, known as joeys, are fully weaned around a year old. Koalas have few natural predators and parasites, but are threatened by various pathogens, such as Chlamydiaceae bacteria and koala retrovirus.\"",
        "response": "Eucalyptus makes up most of their diet. Joeys spend the first six to seven months of their lives in their mothers' pouches. They are threatened by pathogens like Chlamydiaceae and koala retrovirus.",
        "prompt_slug": "closed_qa"
    },
    {
        "query": "Where is the Porce III Dam",
        "context": "\"The Porce III Dam is an embankment dam on the Porce River 90 kilometres (56 mi) northeast of Medell\\u00edn in Antioquia Department, Colombia. The dam was constructed between 2004 and 2011 for the primary purpose of hydroelectric power generation.\\n\\nBackground\\nBetween 1974 and 1976, hydrological studies were carried out on the Porce River and between 1982 and 1983, studies regarding the river's hydroelectric development were completed. The study recommended the five projects, Porce I, Porce II, Porce III, Porce IV and Ermita\\u00f1o. In December 1984, the feasibility report for Porce III was submitted and complementary studies were carried out between 1986 and 1996. In 2002, the design and consultancy contracts were awarded along with the environmental license issued. In 2004, construction on the dam began and the river was diverted by 2007. By 2010, the dam began to impound the reservoir and was complete by 2010. Between 2011, all four generators were commissioned.\"",
        "response": "The Porce III Dam is an embankment dam on the Porce River 90 kilometres (56 mi) northeast of Medellín in Antioquia Department, Colombia. The dam was constructed between 2004 and 2011 for the primary purpose of hydroelectric power generation. It is now one of the largest electricity generation plants in Colombia, with a capacity of 660 megawatts.",
        "prompt_slug": "closed_qa/ungrounded_response"
    },
    {
        "query": "According to this reference text about UVA, what years have UVA women won NCAA titles in rowing?",
        "context": "\"In the 21st century alone, UVA teams have won seventeen NCAA championships. The men's teams have won recent NCAA titles in basketball (2019); lacrosse (2003, 2006, 2011, 2019, and 2021); baseball (2015); soccer (2009 and 2014); and tennis (2013, 2015, 2016, 2017, and 2022). UVA women have won recent NCAA titles in rowing (2010 and 2012) and swimming & diving (2021 and 2022). The Cavaliers rank first in the ACC (a power conference) with 22 men's NCAA Championships, and rank second in the conference with nine women's NCAA Championships.\"",
        "response": "Additionally, the UVA women's rowing team has consistently ranked among the top contenders in NCAA rowing competitions.",
        "prompt_slug": "closed_qa/irrelevant_response"
    },
    {
        "query": "Which team drafted Marques Pope?",
        "context": "\"Marquez Phillips Pope (born October 29, 1970 in Nashville, Tennessee) is a former professional American football player who was drafted by the San Diego Chargers in the 2nd round (33rd overall) of the 1992 NFL Draft. A 5'11\\\" cornerback-safety from Fresno State University, Pope played for 5 teams in 10 NFL seasons from 1992 to 2001. His best year as a pro came during the 1996 season for the San Francisco 49ers, intercepting 6 passes with 1 touchdown. During his pro career, Pope was known to be among the hardest hitters in the NFL.\"",
        "response": "The team that drafted Marquez Pope was the San Diego Chargers in the 1992 NFL Draft.",
        "prompt_slug": "closed_qa"
    },
    {
        "query": "Given the text about the 2019 Masters Tournament, tell me who won the golf tournament, where the tournament was held and what age the winner was?",
        "context": "\"The 2019 Masters Tournament was the 83rd edition of the Masters Tournament and the first of golf's four major championships in 2019, held between April 11 and 14 at Augusta National Golf Club in Augusta, Georgia. It left him one shy of Jack Nicklaus's record six Masters wins, and three short of his record eighteen major wins. It was 11 years after his last major win, the 2008 U.S. Open, and 14 years since his last Masters win beating the previous record of 13 years held by Gary Player. Due to the high profile of Woods, and his storied fall from the top of the game due to personal issues and injuries, the victory generated a large amount of publicity around the world, and is regarded as one of the great sporting comebacks.\"",
        "response": "Tiger Woods won the golf tournament at Augusta National Golf Club in Augusta, GA and was 43 years old.",
        "prompt_slug": "closed_qa/irrelevant_context"
    },
{
        "query": "Who wrote the music and lyrics for Replay?",
        "context": "\"\\\"Tomorrow\\\" is a show tune from the musical Annie, with music by Charles Strouse and lyrics by Martin Charnin, published in 1977. The number was originally written as \\\"Replay\\\" (The Way We Live Now) for the 1970 short film Replay, with both music and lyrics by Strouse.\\n\\nIn 1972 Strouse, and Thomas Meehan began working on a musical based on the Little Orphan Annie comic strips, Annie. Strouse contributed a reworked version of \\\"Replay\\\" as \\\"Tomorrow\\\". \\\"Annie\\\" took until 1976 to reach Broadway and during that time Strouse reworked \\\"Replay\\\" again for possible use in what became \\\"Charlie and Algernon\\\", a musical version of Daniel Keyes's Flowers for Algernon with book by playwright David Rogers although it was not ultimately used. \\\"\"",
        "response": "Charles Strouse",
        "prompt_slug": "closed_qa"
    },
    {
        "query": "What is Sinking Sand?",
        "context": "\"Quicksand, also known as sinking sand, is a colloid consisting of fine granular material (such as sand, silt or clay) and water. It forms in saturated loose sand when the sand is suddenly agitated. When water in the sand cannot escape, it creates a liquefied soil that loses strength and cannot support weight. Quicksand can form in standing water or in upward flowing water (as from an artesian spring). In the case of upward flowing water, forces oppose the force of gravity and suspend the soil particles.\"",
        "response": "Sinking Sand is most commonly known as Quicksand.",
        "prompt_slug": "closed_qa"
    },
    {
        "query": "What book was the musical Hamilton based on?",
        "context": "\"Hamilton is a sung-and-rapped-through musical with music, lyrics, and a book by Lin-Manuel Miranda, based on the 2004 book Alexander Hamilton by Ron Chernow. The musical tells the story of American Founding Father Alexander Hamilton. Composed over a seven-year period from 2008 to 2015, Miranda says Hamilton was originally a hip hop concept album in his head. The show draws heavily from hip hop, as well as R&B, pop, soul, and traditional-style show tunes. It casts non-white actors as the Founding Fathers of the United States and other historical figures. Miranda described Hamilton as about \\\"America then, as told by America now.\\\"\"",
        "response": "Hamilton, the musical was based on the 2004 book Alexander Hamilton by Ron Chernow.",
        "prompt_slug": "closed_qa"
    },
    {
        "query": "When he lived in the mountains of Alaska, was Richard Proenneke in full autonomy?",
        "context": "\"Richard Louis Proenneke (/\\u02c8pr\\u025bn\\u0259ki\\u02d0/; May 4, 1916 \\u2013 April 20, 2003) was an American self-educated naturalist, conservationist, writer, and wildlife photographer who, from the age of about 51, lived alone for nearly thirty years (1968\\u20131998) in the mountains of Alaska in a log cabin that he constructed by hand near the shore of Twin Lakes. Proenneke hunted, fished, raised and gathered much of his own food, and also had supplies flown in occasionally. He documented his activities in journals and on film, and also recorded valuable meteorological and natural data. The journals and film were later used by others to write books and produce documentaries about his time in the wilderness.\"",
        "response": "No, he was getting some of his food by plane.",
        "prompt_slug": "closed_qa"
    },
    {
        "query": "Given this paragraph about the history of the Calgary Flames hockey team, what year did the Calgary Flames win their only Stanley Cup",
        "context": "\"The team was founded in 1972 in Atlanta as the Atlanta Flames before relocating to Calgary in 1980. The Flames played their first three seasons in Calgary at the Stampede Corral before moving into the Scotiabank Saddledome (originally the Olympic Saddledome) in 1983. In 1985\\u201386, the Flames became the first Calgary team since the 1923\\u201324 Tigers to compete for the Stanley Cup. In 1988\\u201389, the Flames won their first and only Stanley Cup title. The Flames' unexpected run to the 2004 Stanley Cup Finals gave rise to the Red Mile, and in 2011, the team hosted and won the second Heritage Classic outdoor game.\"",
        "response": "The Calgary Flames won their only Stanley Cup in 1989",
        "prompt_slug": "closed_qa"
    }
]