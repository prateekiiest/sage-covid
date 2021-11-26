import pandas as pd 
import xml.etree.ElementTree as et 
import sys

usage = """
	Usage: python convert_xml_to_data_frame_csv.py <xml filename> <csv filename> 
"""

def parse_XML(xml_file, df_cols): 
    """Parse the input XML file and store the result in a pandas 
    DataFrame with the given columns. 
    
    The first element of df_cols is supposed to be the identifier 
    variable, which is an attribute of each node element in the 
    XML data; other features will be parsed from the text content 
    of each sub-element. 
    """
    
    xtree = et.parse(xml_file)
    xroot = xtree.getroot()
    rows = []
    
    for node in xroot: 
        res = []
        res.append(node.attrib.get(df_cols[0]))
        for el in df_cols[1:]: 
        #for el in df_cols[0:]:
            if node is not None and node.find(el) is not None:
                res.append(node.find(el).text)
            else: 
                res.append(None)
        rows.append({df_cols[i]: res[i] 
                     for i, _ in enumerate(df_cols)})
    
    out_df = pd.DataFrame(rows, columns=df_cols)
        
    return out_df

if __name__ == "__main__":

	if (len(sys.argv) != 3):
		sys.stderr.write(usage)
		sys.exit(1)

	xml_filename = sys.argv[1]
	csv_filename = sys.argv[2]

	#df_cols = ["PharmacologicalAction", "DescriptorName", "RecordName"]
        df_cols = ["DescriptorName", "RecordName"]
	#df = pd.DataFrame(columns=df_cols)

	etree = et.parse(xml_filename)
	eroot = etree.getroot()

	rows = []

	for elem in eroot.iter("PharmacologicalAction"):

		for elem_2 in elem:

			if ("DescriptorReferredTo" in elem_2.tag):

				for elem_3 in elem_2:
					if ("DescriptorName" in elem_3.tag):
						for elem_4 in elem_3:
							descriptor_name = elem_4.text

			if ("PharmacologicalActionSubstanceList" in elem_2.tag):
				for elem_3 in elem_2:
					if ("Substance" in elem_3.tag):
						for elem_4 in elem_3:
							if ("RecordName" in elem_4.tag):
								for elem_5 in elem_4:
									record_name = elem_5.text
									rows.append([descriptor_name, record_name])

	#print(rows)									
	df = pd.DataFrame(rows, columns=df_cols)
	df.to_csv(csv_filename)

		
