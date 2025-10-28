
#masking starting parts
mask_start = 'Your role is to ANONYMIZE all text that is provided by the user. '
mask_start_iter = 'Your role is to ANONYMIZE all text that is provided by the user. You have already tried to anonymize the following text and failed, you left details which were used to identify the text by an expert. Be more agressive in your efforts to anonymize and pay more attention to not leave even the most minute details. '

#masking middle parts
mask_name = 'The text you need to anonymize will be about the firm: '
mask_ticker = 'whose ticker is: '
mask_industry = 'and is in the industry of: '

#masking end parts
mask_end = 'Any instance of the firm, their products, the industry or industry related areas they are involved in should be anonymized. Absolutely no mention of key words related to that industry can be left in the text. After you have anonymized a text, NOBODY, not even an expert financial analyst should be able to use the text to identify the company or the industry the company operates in. For example, if the text is: The country’s largest phone producer Apple had great phone-related earnings but Google did not in 2024, likely because of Apple’s slogan Think Different, then you should ANONYMIZE it to: The country’s largest product_type_1 producer Company_1 had great product_type_1 related earnings but Company_2 did not in time_1 likely because of Company_1’s slogan slogan_1. You should also ANONYMIZE any other information which one could use to identify the company or to make an educated guess about its identity. Stock tickers are identifiers, are usually four or fewer capitalized letters (consider TIKR as a stand-in for an arbitrary ticker), and can be referenced in the text in the following formats; SYMBOL:TIKR, $TIKR, >TIKR, $ TIKR, SYMBOL TIKR, SYMBOL: TIKR, > TIKR. Make sure you ANONYMIZE TIKR to ticker_x. Also ANONYMIZE any other identifiers related to companies including the names of individuals, locations, industries, sectors, product names and types and generic product lines, services, times, years, dates and all numbers and percentages in the text, including units. These should be replaced with name_x, location_x, industry_x, sector_x, product_x, product_type_x, product_line_x, service_x, time_x, year_x, date_x and number a, b, c, respectively. Also replace any website or internet links with link_x. You should never just delete an identifier; instead, always replace it with an anonymized analog. After you read and ANONYMIZE the text, you should output the anonymized text and nothing else.'
mask_end_iter = 'Any instance of the firm, their products, the industry or industry related areas they are involved in should be anonymized. Absolutely no mention of key words related to that industry can be left in the text. After you have anonymized a text, NOBODY, not even an expert financial analyst should be able to use the text to identify the company or the industry the company operates in. For example, if the text is: The country’s largest phone producer Apple had great phone-related earnings but Google did not in 2024, likely because of Apple’s slogan Think Different, then you should ANONYMIZE it to: The country’s largest product_type_1 producer Company_1 had great product_type_1 related earnings but Company_2 did not in time_1 likely because of Company_1’s slogan slogan_1. You should also ANONYMIZE any other information which one could use to identify the company or to make an educated guess about its identity. Stock tickers are identifiers, are usually four or fewer capitalized letters (consider TIKR as a stand-in for an arbitrary ticker), and can be referenced in the text in the following formats; SYMBOL:TIKR, $TIKR, >TIKR, $ TIKR, SYMBOL TIKR, SYMBOL: TIKR, > TIKR. Make sure you ANONYMIZE TIKR to ticker_x. Also ANONYMIZE any other identifiers related to companies including the names of individuals, locations, industries, sectors, product names and types and generic product lines, services, times, years, dates and all numbers and percentages in the text, including units. These should be replaced with name_x, location_x, industry_x, sector_x, product_x, product_type_x, product_line_x, service_x, time_x, year_x, date_x and number a, b, c, respectively. Also replace any website or internet links with link_x. You should never just delete an identifier; instead, always replace it with an anonymized analog. You have already failed me before, take no chances this time and be aggressive in censoring the text. After you read and ANONYMIZE the text, you should output the anonymized text and nothing else.'

mask_template = {
    'template': [ # must contain 'template' in structure
        mask_start,
        {'column': 'COMNAM', 'prefix': mask_name, 'suffix': ' '}, #COMNAM is Company Name
        {'column': 'Companies', 'prefix': mask_ticker, 'suffix': ' '}, #Companies is the Ticker of the firm
        {'column': 'SIC_Industry', 'prefix': mask_industry, 'suffix': '.'}, #SIC_Industry is the industry of the firm
        mask_end
    ]
}
mask_template_iter = {
    'template': [ # must contain 'template' in structure
        mask_start_iter,
        {'column': 'COMNAM', 'prefix': mask_name, 'suffix': ' '}, #COMNAM is Company Name
        {'column': 'Companies', 'prefix': mask_ticker, 'suffix': ' '}, #Companies is the Ticker of the firm
        {'column': 'SIC_Industry', 'prefix': mask_industry, 'suffix': '.'}, #SIC_Industry is the industry of the firm
        mask_end_iter
    ]
}


#paraphrasing starting parts
para_start = 'You are to paraphrase text ensuring that the core info is unchanged such that someone who has memorized the text could not recognize it. Make sure to replace all unique or identifying words and phrases. As well change the sentence structure so that the resulting text is completely different to the original. This includes changing quotes and other highly recognizable structures in the text. '
para_start_iter = 'You are to paraphrase text ensuring that the core info is unchanged such that someone who has memorized the text could not recognize it. You have already failed to paraphrase the next such that it is anonymous. Be more aggressive in your paraphrasing and anonymizing of the text. Make sure to replace all unique or identifying words and phrases. As well change the sentence structure so that the resulting text is completely different to the original. This includes changing quotes and other highly recognizable structures in the text. '

#paraphrasing middle parts
para_name = 'The text provided will be about the firm: '
para_ticker = ' whose ticker is: '
para_industry = ' and is in the industry: '

#paraphrasing end parts
para_end = ' Any information which could ever be used to identify the firm the text is about or when the text was written MUST be anonymized. Absolutely no mention of key words related to that industry or any other industries can be left in the text. If you observe any names, proper nouns, aliases, acronyms, dates, industry references (related and unrelated to the industry of the firm in question), government agency names, product names or types, etc., replace them with anonymous strings. Anonymity is the prime objective, leave zero hints for someone who has memorized the text! Ensure that the paraphrased text has approximately the same number of words as the original text. After you read and ANONYMIZE the text, you should output the anonymized text and nothing else.'
para_end_iter = ' Any information which could ever be used to identify the firm the text is about or when the text was written MUST be anonymized. Absolutely no mention of key words related to that industry or any other industries can be left in the text. If you observe any names, proper nouns, aliases, acronyms, dates, industry references (related and unrelated to the industry of the firm in question), government agency names, product names or types, etc., replace them with anonymous strings. Anonymity is the prime objective, leave zero hints for someone who has memorized the text! Ensure that the paraphrased text has approximately the same number of words as the original text. You have already failed once before, take no chances this time, be aggressive in your effort to censor text. After you read and ANONYMIZE the text, you should output the anonymized text and nothing else.'

para_template = {
    'template': [ # must contain 'template' in structure
        para_start,
        {'column': 'COMNAM', 'prefix': para_name, 'suffix': ' ,'}, #COMNAM is Company Name
        {'column': 'Companies', 'prefix': para_ticker, 'suffix': ' ,'}, #Companies is the Ticker of the firm
        {'column': 'SIC_Industry', 'prefix': para_industry, 'suffix': '.'}, #SIC_Industry is the industry of the firm
        para_end
    ]
}

para_template_iter = {
    'template': [ # must contain 'template' in structure
        para_start_iter,
        {'column': 'COMNAM', 'prefix': para_name, 'suffix': ' ,'}, #COMNAM is Company Name
        {'column': 'Companies', 'prefix': para_ticker, 'suffix': ' ,'}, #Companies is the Ticker of the firm
        {'column': 'SIC_Industry', 'prefix': para_industry, 'suffix': '.'}, #SIC_Industry is the industry of the firm
        para_end_iter
    ]
}



#Guessing Test
de_neuter_name = "You will receive a body of text which has been anonymized. You are omniscient. Use all of your knowledge and any clues in the text to identify which company the text is about, its industry, and the year it was written in. Make your best guess if you are unsure. Your options for industry guesses are the following: Aerospace, Agriculture, Automobiles and Trucks, Banking, Beer and Liquor, Construction Materials, Printing and Publishing, Shipping Containers, Business Services, Chemicals, Computers, Apparel, Construction, Coal, Pharmaceutical Products, Electrical Equipment, Fabricated Products, Financial Services, Food Products, Recreation, Precious Metals, Defense, Computer Hardware, Healthcare, Consumer Goods, Insurance, Measuring and Control Equipment, Machinery, Restaurants, Hotels, and Motels, Medical Equipment, Non-Metallic and Industrial Mining, Petroleum and Natural Gas, Paper, Personal Services, Real Estate, Retail, Rubber and Plastic Products, Shipbuilding, Railroad Equipment, Tobacco Products, Candy and Soda, Software, Steel Works, Communication, Entertainment, Transportation, Textiles, Utilities, Wholesale. Provide your estimate EXACTLY in the following format, with NO other text at all (TIKR is your estimate of the ticker, NAME is your estimate of the firm name, IND is your estimate of the firm's industry, YYYY-MM-DD is your estimate of the date): **Ticker Estimate: TIKR**,**Name Estimate: NAME**,**Industry Estimate: IND**,**Date Estimate: YYYY-MM-DD**"

#Sentiment Extraction
sentiment = 'You are an expert Financial Analyst. Your task is to read news articles and infer if the reported news is a good or bad signal for the future returns of the security being written about. Based on your answer, I will choose to buy or sell that security. After reading the news, provide a bull or bear signal as two numbers: the direction and the magnitude. For direction, output only 0,1, or NA, where 0 is bearish, 1 is bullish, and NA means there is no information relevant to security prices. For magnitude, output a number ranging from 0 to 1 ONLY, where numbers near 0 indicate slightly bearish (if the direction is 0) or slightly bullish (if the direction is 1), and numbers near 1 indicate highly bearish (if the direction is 0) or highly bullish (if the direction is 1). Provide your output exactly in the following format, with no other text at all: **Direction Estimate: DIRECTION**,**Magnitude Estimate: MAGNITUDE**'




