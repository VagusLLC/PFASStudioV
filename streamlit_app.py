# streamlit_app.py

import streamlit as st
import pymongo
import gridfs
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import zlib
import pandas as pd
from bson.objectid import ObjectId
from streamlit_plotly_events import plotly_events
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
# Initialize connection.
st.set_page_config(layout="wide")

# Uses st.cache_resource to only run once.
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(**st.secrets["mongo"])

client = init_connection()
db = client.fireworks
filepad = db["filepad"]
fs = gridfs.GridFS(client.fireworks, "filepad_gfs")
# Uses st.cache_resource to only run once.
@st.cache_resource
def get_data():
    db = client.fireworks
    items = db.workflows.find({"state":"COMPLETED"})
    items = list(items)  # make hashable for st.cache_data
    return items

def get_files(name):    
    filename = "xtbopt_xyz_"+name
    doc = filepad.find_one({"identifier": filename})
    gfs_id = doc["gfs_id"]
    file_contents = fs.get(ObjectId(gfs_id)).read()
    xyz = zlib.decompress(file_contents)

    filename = "xtbopt_pdb_"+name
    doc = filepad.find_one({"identifier": filename})
    gfs_id = doc["gfs_id"]
    file_contents = fs.get(ObjectId(gfs_id)).read()
    pdb = zlib.decompress(file_contents)

    try:
        filename = "LUMO_"+name
        doc = filepad.find_one({"identifier": filename})
        gfs_id = doc["gfs_id"]
        file_contents = fs.get(ObjectId(gfs_id)).read()
        lumo = zlib.decompress(file_contents)

        filename = "HOMO_"+name
        doc = filepad.find_one({"identifier": filename})
        gfs_id = doc["gfs_id"]
        file_contents = fs.get(ObjectId(gfs_id)).read()
        homo = zlib.decompress(file_contents)

        filename = "ESP_"+name
        doc = filepad.find_one({"identifier": filename})
        gfs_id = doc["gfs_id"]
        file_contents = fs.get(ObjectId(gfs_id)).read()
        esp = zlib.decompress(file_contents)
    except:
        homo = None
        lumo = None
        esp = None
    return xyz, pdb, homo, lumo, esp

def process_pdb(pdb):
    pdb = pdb.decode("utf-8")
    pdb = pdb.split("\n")
    # Get all hetatm lines
    hetatms = [line for line in pdb if line.startswith("HETATM")]
    # Get all connect lines
    connects = [line for line in pdb if line.startswith("CONECT")]
    
def draw_smiles(smiles,elem_id = "id1", height = "100%", scale = 1):
    return f"""
        <img id={elem_id} data-smiles-options="{{align-items: "center" }}" height={height}/>\
        <script type="text/javascript" src="https://unpkg.com/smiles-drawer@2.0.1/dist/smiles-drawer.min.js"></script>
        <script>
            let moleculeOptions_{elem_id} = {{
                "scale": {scale},
                "bondThickness": 1,
                "shortBondLength": 0.8,
                "bondSpacing": 5.1000000000000005,
                "atomVisualization": "default",
                "isomeric": true,
                "debug": false,
                "terminalCarbons": true,
                "explicitHydrogens": true,
                "overlapSensitivity": 0.42,
                "overlapResolutionIterations": 1,
                "compactDrawing": false,
                "fontFamily": "Arial, Helvetica, sans-serif",
                "fontSizeLarge": 11,
                "fontSizeSmall": 3,
                "padding": 2,
                "experimentalSSSR": true,
                "kkThreshold": 0.1,
                "kkInnerThreshold": 0.1,
                "kkMaxIteration": 20000,
                "kkMaxInnerIteration": 50,
                "kkMaxEnergy": 1000000000,
                "themes": {{
                    "light": {{
                        "C": "#222",
                        "O": "#e74c3c",
                        "N": "#3498db",
                        "F": "#27ae60",
                        "CL": "#16a085",
                        "BR": "#d35400",
                        "I": "#8e44ad",
                        "P": "#d35400",
                        "S": "#f1c40f",
                        "B": "#e67e22",
                        "SI": "#e67e22",
                        "H": "#666",
                        "BACKGROUND": "#fff"
                    }}
                }}
            }};
            let sd_{elem_id} = new SmiDrawer(moleculeOptions_{elem_id}, {{}});
            sd_{elem_id}.draw('{smiles}', '#{elem_id}')
        </script>
        """

st.header("PFAS Studio V by Vagus, LLC", divider=True)
items = get_data()
props = [item['scalar_properties'] for item in items]
tensor_props = [item['tensor_properties'] for item in items]

# Get fp, fm, f0 keys as dict
fukui_props = [{k: v for k, v in item.items() if k in ['[fp]', '[fm]', '[f0]']} for item in tensor_props]

partial_charges = [[v for k, v in item.items() if k in ['Partial Charge [e]']] for item in tensor_props]

for x, item in zip(enumerate(props), items):
    props[x[0]]['Smiles'] = item['metadata']['smiles']
keys = props[0].keys()
data = {k: [prop[k] for prop in props] for k in keys}
data['name'] = [item['name'] for item in items]
names = ["CASRN: "+item['name'] for item in items]
prop_list = [k for k in data.keys() if k not in ['Smiles', 'name']]

cc1, cc2, cc3 = st.columns([0.25, 0.5, 0.25])
with cc3:
    tt1, tt2 = st.tabs(["PFAS Dataset", "PFAS Similarity"])
    with tt1:
        x = st.selectbox('X-Axis-new', prop_list)
        y = st.selectbox('Y-Axis-new', prop_list)
        # color = st.selectbox('Color by', prop_list)
        fig = px.scatter(
            data,
            x = x,
            y = y,
            # color = color,
            hover_name = names,
            # hover_data = props[0].keys(),
            )
        f = go.Figure()
        f.add_trace(go.Histogram2dContour(
            x = data[x],
            y = data[y],
            colorscale = 'Teal',
            reversescale=False,
            showscale=False,
            hoverinfo='skip',
            xaxis = 'x',
            yaxis = 'y'
        ))
        f.add_trace(go.Scatter(
            x = data[x],
            y = data[y],
            xaxis = 'x',
            yaxis = 'y',
            mode = 'markers',
            text=names,
            name='',
            hovertemplate="%{text}",
            marker = dict(
                color = 'rgba(0,0,0,1.0)',
                size = 3
            )
        ))

        f.update_layout(
            autosize = False,
            xaxis = dict(
                zeroline = False,
                domain = [0,0.85],
                showgrid = False,
                title = x
            ),
            yaxis = dict(
                zeroline = False,
                domain = [0,0.85],
                showgrid = False,
                title = y
            ),
            margin = dict(
                b = 80, 
                l = 80, 
                t = 10,
                r = 20
            ),
            width = 450,
            hovermode = 'closest',
            showlegend = False
        )
        selected_points = plotly_events(f, override_height=700)

        if len(selected_points) > 0:
            index = selected_points[0]["pointIndex"]
            casrn = items[selected_points[0]["pointIndex"]]["name"]
            mol_props = props[selected_points[0]["pointIndex"]]
            smiles = data['Smiles'][selected_points[0]["pointIndex"]]
        else:
            index = 0
            casrn = items[0]["name"]
            mol_props = props[0]
            smiles = data['Smiles'][0]
        xyz, pdb, homo, lumo, esp = get_files(casrn)   
    with tt2:
        fp_methods = {"Morgan Fingerprints": AllChem.GetMorganGenerator(), 
         "RDKit Fingerprints": AllChem.GetRDKitFPGenerator(),
         "Atom Pair Fingerprints": AllChem.GetAtomPairGenerator(),
         "Topological Torsion Fingerprints":AllChem.GetTopologicalTorsionGenerator()
        }
        fingerprint = st.selectbox("Fingerprint Method", list(fp_methods.keys()))

        metric = st.selectbox("Similarity Metric", list(map(lambda x: x[0], DataStructs.similarityFunctions)))
        metric_func =list(filter(lambda x: x[0] == metric, DataStructs.similarityFunctions))[0][1]
        N = st.number_input("Top N: ", min_value=1, max_value=100, value=10, step=1)
        ms = [Chem.MolFromSmiles(data['Smiles'][x]) for x in range(len(data['Smiles']))]
        fpgen = fp_methods[fingerprint]
        fps = [fpgen.GetFingerprint(m) for m in ms]
        # calculate TanimotoSimilarity for all indices expect index in fps
        indices = [x for x in range(len(fps)) if x != index]
        sim = [DataStructs.FingerprintSimilarity(fps[index], fps[x], metric = metric_func) for x in indices]

        sim_names = [data['name'][x] for x in indices]
        sim_smiles = [data['Smiles'][x] for x in indices]
        # return top N similar molecules
        # N = 10
        topN = sorted(zip(sim, sim_names, sim_smiles), reverse=True)[:N]
        df = pd.DataFrame({"Similarity": [x[0] for x in topN], "CASRN": [x[1] for x in topN]})
        st.download_button("Press to Download List", df.to_csv(index=False).encode("utf-8"), "PFAS_Similarity.csv", "text/csv", key='download-csv')
        html = f""" 
            <style> 
                .dataframe {{   
                    border-collapse: collapse;  
                    margin: 0px 0;  
                    font-size: 0.9em;   
                    font-family: "Source Sans Pro",sans-serif;  
                    min-width: 100%;    
                    border-radius: 15px;    
                    border-spacing: 30px;   
                    padding: 5px 5px 5px 5px;   
                    overflow:hidden;    
                }}  
                .dataframe thead th {{  
                    background-color: #d4edf7;  
                    font-family: "Source Sans Pro",sans-serif;  
                    color: #000000; 
                    text-align: center; 
                }}  
                .dataframe thead th:nth-child(1) {{ 
                    background-color: #d4edf7;  
                    font-family: "Source Sans Pro",sans-serif;  
                    color: #000000; 
                    text-align: left;   
                }}  
                .dataframe tbody td:nth-child(2){{  
                    text-align:center;  
                }}  
                .dataframe thead th,td {{   
                    padding: 10px   
                }}  
                .dataframe tbody td{{   
                    font-family: sans-serif;    
                    color: #000000; 
                    text-align: left;   
                    padding: 10px;  
                }}  
                .dataframe tbody tr {{  
                    border-bottom: 1px solid #dddddd;   
                    text-align: left;   
                }}  
                .dataframe tbody tr:nth-of-type(even) {{    
                    background-color: #f3f3f3;  
                }}  
                h1 {{   
                    font-family: "Source Sans Pro",sans-serif;  
                }}  
            </style>    
            <center>    
            <table class = "dataframe"> 
            <thead >    
            <tr>    
                <th>Similarity</th> 
                <th>CASRN</th>  
                <th>Smiles</th>
            </tr>
            </thead>
            <tbody>
        """+''.join(list(map(lambda i: f"""
            <tr>
                <td >
                    {i[1][0]:.4f}
                </td>
                <td>
                    {i[1][1]}
                </td>
                <td >
                    <center>{draw_smiles(i[1][2], elem_id = "id"+str(i[0]), scale = 0.3)}</center>
                </td>
            </tr>
        """, zip(range(N), topN))))+f"""
            </tbody>
            </table>
            </center>
        """
        st.components.v1.html(html, height = 800)   
with cc1:
    structure, ir_tab, local_mode_tab = st.tabs(["3D Structure", "IR Properties", "Local Mode"])
    with structure:
        opt = st.selectbox('3D Views', ["Structure", "Fukui Indices", "Partial Charges", "HOMO-LUMO Orbitals", "Electrostatic Potential"])
        if opt == "Structure":
            html = """
                <script src="https://3Dmol.org/build/3Dmol-min.js"></script>                     
                <style>
                    .mol-container {
                    width: 100%;
                    height: 400px;
                    position: relative;
                    }
                </style>
                <div id="container-02" class="mol-container"></div>
                <script>
                    let element = 'container-02';
                    let config = {};
                    let labelSpec = {
                        backgroundColor:"white",
                        fontColor:"black",
                        backgroundOpacity:0.75,
                        inFront:true,
                    };
                    let v = $3Dmol.createViewer( element, config );
                    var m = v.addModel(`"""+pdb.decode("utf-8")+"""`, "pdb", {keepH:true, assignBonds:true});
                    v.setBackgroundColor(0xffffff, 0.0);
                    v.setStyle({},{cartoon:{}, sphere:{scale:0.25, colorscheme:'Jmol'}, stick:{radius:0.15, colorscheme:'Jmol'}});
                    v.animate({loop: "forward",reps: 1});
                    v.zoomTo();                                      /* set camera */
                    v.render();                                      /* render scene */
                </script>"""
            st.components.v1.html(html, height = 425)
        elif opt == "Fukui Indices":
            t1, t2, t3 = st.tabs(["f(+)","f(-)","f(0)"])
            with t1:
                html = """
                    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>                     
                    <style>
                        .mol-container {
                            width: 100%;
                            height: 400px;
                            position: relative;
                        }
                    </style>
                    <div id="container-02" class="mol-container"></div>
                    <script>
                        let element = 'container-02';
                        let config = {};
                        let labelSpec = {
                            backgroundColor:"green",
                            fontColor:"white",
                            backgroundOpacity:0.75,
                            inFront:true,
                        };
                        let v = $3Dmol.createViewer( element, config );
                        var m = v.addModel(`"""+pdb.decode("utf-8")+"""`,"pdb", {keepH:true, assignBonds:true});
                        v.setBackgroundColor(0xffffff, 0.0);
                        """+''.join(list(map(lambda i: f"""v.addLabel("{i[1]}", labelSpec, {{index: {i[0]}}});""", enumerate(fukui_props[index]["[fp]"]))))+"""
                        v.setStyle({},{cartoon:{}, sphere:{scale:0.25, colorscheme:'Jmol'}, stick:{radius:0.15, colorscheme:'Jmol'}});
                        v.zoomTo();                                      /* set camera */
                        v.render();                                      /* render scene */
                    </script>
                    """
                st.components.v1.html(html, height = 425)
            with t2:
                html = """
                        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>                     
                        <style>
                            .mol-container {
                            width: 100%;
                            height: 400px;
                            position: relative;
                            }
                        </style>
                        <div id="container-02" class="mol-container"></div>
                        <script>
                            let element = 'container-02';
                            let config = {};
                            let labelSpec = {
                                backgroundColor:"red",
                                fontColor:"black",
                                backgroundOpacity:0.5,
                                inFront:true,
                            };
                            let v = $3Dmol.createViewer( element, config );
                            var m = v.addModel(`"""+pdb.decode("utf-8")+"""`, "pdb", {keepH:true, assignBonds:true});
                            v.setBackgroundColor(0xffffff, 0.0);
                        """+''.join(list(map(lambda i: f"""v.addLabel("{i[1]}", labelSpec, {{index: {i[0]}}});""", enumerate(fukui_props[index]["[fm]"]))))+"""
                        v.setStyle({},{cartoon:{}, sphere:{scale:0.25, colorscheme:'Jmol'}, stick:{radius:0.15, colorscheme:'Jmol'}});
                            v.zoomTo();                                      /* set camera */
                            v.render();                                      /* render scene */
                        </script>
                    """
                st.components.v1.html(html, height = 425)
            with t3:
                html = """
                        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>                     
                        <style>
                            .mol-container {
                            width: 100%;
                            height: 400px;
                            position: relative;
                            }
                        </style>
                        <div id="container-02" class="mol-container"></div>
                        <script>
                            let element = 'container-02';
                            let config = {};
                            let labelSpec = {
                                backgroundColor:"white",
                                fontColor:"black",
                                backgroundOpacity:0.75,
                                inFront:true,
                            };
                            let v = $3Dmol.createViewer( element, config );
                            var m = v.addModel(`"""+pdb.decode("utf-8")+"""`, "pdb", {keepH:true, assignBonds:true});
                            v.setBackgroundColor(0xffffff, 0.0);
                        """+''.join(list(map(lambda i: f"""v.addLabel("{i[1]}", labelSpec, {{index: {i[0]}}});""", enumerate(fukui_props[index]["[fp]"]))))+"""
                        v.setStyle({},{cartoon:{}, sphere:{scale:0.25, colorscheme:'Jmol'}, stick:{radius:0.15, colorscheme:'Jmol'}});
                            v.zoomTo();                                      /* set camera */
                            v.render();                                      /* render scene */
                        </script>
                    """
                st.components.v1.html(html, height = 425)
        elif opt == "Partial Charges":
            html = """
                <script src="https://3Dmol.org/build/3Dmol-min.js"></script>                     
                <style>
                    .mol-container {
                    width: 100%;
                    height: 400px;
                    position: relative;
                    }
                </style>
                <div id="container-02" class="mol-container"></div>
                <script>
                    let element = 'container-02';
                    let config = {};
                    let labelSpec = {
                        backgroundColor:"white",
                        fontColor:"black",
                        backgroundOpacity:0.75,
                        inFront:true,
                    };
                    let v = $3Dmol.createViewer( element, config );
                    var m = v.addModel(`"""+pdb.decode("utf-8")+"""`, "pdb", {keepH:true, assignBonds:true});
                    v.setBackgroundColor(0xffffff, 0.0);
                """+''.join(list(map(lambda i: f"""v.addLabel("{i[1]:.3f}", labelSpec, {{index: {i[0]}}});""", enumerate(partial_charges[index][0]))))+"""
                    v.setStyle({},{cartoon:{}, sphere:{scale:0.25, colorscheme:'Jmol'}, stick:{radius:0.15, colorscheme:'Jmol'}});
                        v.vibrate(10, 1);
                        v.animate({loop: "forward",reps: 1});
                        v.zoomTo();                                      /* set camera */
                        v.render();                                      /* render scene */
                </script>"""
            st.components.v1.html(html, height = 425)
        elif opt == "HOMO-LUMO Orbitals":
            level = st.slider("Isosurface Value", min_value=0.0, max_value=0.1, value=0.001, step=0.001)
            html = """
                <script src="https://3Dmol.org/build/3Dmol-min.js"></script>                     
                <style>
                    .mol-container {
                            width: 400px;
                            height: 312px;
                            position: relative;
                            }
                </style>
                <center>
                <table class = "dataframe">
                <tbody>
                <tr>
                <td>
                <div id="container-homo" class="mol-container"></div>
                </td>
                </tr>
                <tr>
                <td>
                <div id="container-lumo" class="mol-container"></div>
                </td>
                </tr>
                </tbody>
                </table>
                </center>

                <script>
                    let element = 'container-homo';
                    let labelSpec = {
                        alignment:"center",
                        backgroundColor:"white",
                        fontColor:"black",
                        backgroundOpacity:0.5,
                        inFront:true,
                    };
                    let v = $3Dmol.createViewer( element, {} );
                    var m = v.addModel(`"""+pdb.decode("utf-8")+"""`, "pdb", {keepH:true, assignBonds:true});
                    var voldata = new $3Dmol.VolumeData(`"""+homo.decode("utf-8")+"""`, "cube");
	    		    v.addIsosurface(voldata, {isoval:"""+str(level)+""", color:"red", opacity: 0.95});
	    		    v.addIsosurface(voldata, {isoval:"""+str(-level)+""", color:"blue", opacity: 0.95});
                    v.setBackgroundColor(0xffffff, 0.0);
                    v.setStyle({},{cartoon:{}, sphere:{scale:0.25, colorscheme:'Jmol'}, stick:{radius:0.15, colorscheme:'Jmol'}});
                    v.zoomTo();
                    v.render();
                    let element_lumo = 'container-lumo';
                    let v_lumo = $3Dmol.createViewer( element_lumo, {} );
                    var m_lumo = v_lumo.addModel(`"""+pdb.decode("utf-8")+"""`, "pdb", {keepH:true, assignBonds:true});
                    var voldata = new $3Dmol.VolumeData(`"""+lumo.decode("utf-8")+"""`, "cube");
	    		    v_lumo.addIsosurface(voldata, {isoval:"""+str(level)+""", color:"red", opacity: 0.95});
	    		    v_lumo.addIsosurface(voldata, {isoval:"""+str(-level)+""", color:"blue", opacity: 0.95});
                    v_lumo.setBackgroundColor(0xffffff, 0.0);
                    v_lumo.setStyle({},{cartoon:{}, sphere:{scale:0.25, colorscheme:'Jmol'}, stick:{radius:0.15, colorscheme:'Jmol'}});
                    v_lumo.zoomTo();
                    v_lumo.linkViewer(v);
                    v.linkViewer(v_lumo);
                    v_lumo.render();
                </script>
                """
            st.components.v1.html(html, height = 700)
        elif opt == "Electrostatic Potential":
            surface_map = {
                    "van der Waals Surface": "VDW",
                    "Molecular Surface": "MS",
                    "Solvent Accessible Surface": "SAS",
                    "Solvent Exposed Surface": "SES"
                }
            s_type = st.selectbox("Surface Type", ["van der Waals Surface", "Molecular Surface", "Solvent Accessible Surface", "Solvent Exposed Surface"])
            val = st.slider("Max/Min Electrostatic Value", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
            surface_type = surface_map[s_type]
            html = """
                    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>                     
                    <style>
                        .mol-container {
                        width: 100%;
                        height: 550px;
                        position: relative;
                        }
                    </style>
                    <center><div id="container-05" class="mol-container"></div></center>
                    <script>
                        let element = 'container-05';
                        let labelSpec = {
                            alignment:"center",
                            backgroundColor:"white",
                            fontColor:"black",
                            backgroundOpacity:0.5,
                            inFront:true,
                        };
                        let v = $3Dmol.createViewer( element, {} );
                        var m = v.addModel(`"""+pdb.decode("utf-8")+"""`, "pdb", {keepH:true, assignBonds:true});
                        var voldata = new $3Dmol.VolumeData(`"""+esp.decode("utf-8")+"""`, "cube");
                        v.addSurface($3Dmol.SurfaceType."""+surface_type+""", 
                        {opacity: 0.95, 
                        voldata: voldata,
                        volscheme: {gradient: 'rwb', min:"""+str(-val)+""", max:"""+str(val)+"""}})
	    		        v.setBackgroundColor(0xffffff, 0.0);
                        v.setStyle({},{cartoon:{}, sphere:{scale:0.25, colorscheme:'Jmol'}, stick:{radius:0.15, colorscheme:'Jmol'}});
                        v.zoomTo();
                        v.render();
                    </script>
                    """
            st.components.v1.html(html, height = 575)
    with ir_tab: 
        # t1, t2 = st.tabs(["IR Spectra", "Normal Mode Visualization"])
        df = pd.DataFrame({
                "Frequency [cm⁻¹]":tensor_props[index]["Frequency [cm⁻¹]"], 
                "IR Itensity [kJ/mol]":tensor_props[index]["IR Itensity [kmmol⁻¹]"],
                "Reduced Mass [amu]":tensor_props[index]["Reduced Mass [amu]"]})
        # with t1:
            # fig = make_subplots(rows = 1, cols = 2, shared_xaxes=True)
        fig = go.Figure(layout = {'height': 275})
        fig.add_trace(
            go.Bar(x=df["Frequency [cm⁻¹]"], y=df["IR Itensity [kJ/mol]"],  name="IR Itensity [kJ/mol]")
        )

            # fig.add_trace(
            #     go.Bar(x=df["Frequency [cm⁻¹]"], y=df["Reduced Mass [amu]"],    name="Reduced Mass [amu]"),
            #     row = 1, col = 2
            # )

        fig.update_xaxes(title_text="Frequency [cm⁻¹]")

            # Set y-axes titles
        fig.update_yaxes(title_text="Integrated Molar Absorptivity [km/mol]")
            # fig.update_yaxes(title_text="Reduced Mass [amu]",col = 2)  
        fig.update_layout(margin = dict(l = 0, r = 0, t = 25, b = 0))
        fig.update(layout_showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        # with t2:
        freq = st.selectbox('Frequency [cm⁻¹]', df["Frequency [cm⁻¹]"])
        freq_index = df["Frequency [cm⁻¹]"].tolist().index(freq)
        dx = tensor_props[index]["dx [Å]"][freq_index]
        dy = tensor_props[index]["dy [Å]"][freq_index]
        dz = tensor_props[index]["dz [Å]"][freq_index]
        xyz_lines = xyz.decode("utf-8").split("\n")
        xyz_header = xyz_lines[0:2]
        xyz_coords = list(map(lambda x: " ".join(x[1].split())+f" {dx[x[0]]} {dy[x  [0]]} {dz[x[0]]}", enumerate(xyz_lines[2:-1])))
        xyz_vibe = "\n".join(xyz_header+xyz_coords)
        html1 = """
                <script src="https://3Dmol.org/build/3Dmol-min.js"></script>                     
                <style>
                    .mol-container {
                    width: 100%;
                    height: 430px;
                    position: relative;
                    }
                </style>
                <div id="container-022" class="mol-container"></div>
                <script>
                    let element = 'container-022';
                    let config = {};
                    let labelSpec = {
                        alignment:"center",
                        backgroundColor:"white",
                        fontColor:"black",
                        backgroundOpacity:0.5,
                        inFront:true,
                    };
                    let v = $3Dmol.createViewer( element, config );
                var freq_data = `"""+xyz_vibe+"""`
                v.addModel(freq_data, "xyz", {assignBonds:true,});
                v.vibrate(10, 1, true);
                v.animate({'loop': 'backAndForth'});
                v.setStyle({},{sphere:{scale:0.25, colorscheme:'Jmol'}, stick:{radius:0.15, colorscheme:'Jmol'}});
                v.setBackgroundColor(0xffffff, 0.0);
                v.zoomTo();
                v.render(/* no callback */ );
                </script>
            """
        st.components.v1.html(html1, height = 450)
    with local_mode_tab:
        try:
            cols = ["Mode"]
            # list(map(lambda x: cols.append(x), tensor_props[index]["local [modes]"]))
            # st.write(cols)
            df = pd.DataFrame(columns=["Normal Mode", "Local Mode", "Contribution"])
            for i,freq in enumerate(tensor_props[index]["Frequency [cm⁻¹]"]): 
                for j, mode in enumerate(tensor_props[index]["local [modes]"]):
                    df.loc[-1] = [freq] + [mode] + [tensor_props[index]["local [contributions]"][i][j]]
                    df.index = df.index+1
            # st.write(df)
            fig_bar = px.bar(df, x = "Normal Mode", color = "Local Mode", y = "Contribution", barmode='stack')
            # Update ylabel in fig_bar to say "Contribution [%]"
            fig_bar.update_yaxes(title_text="Contribution [%]")
            fig_bar.update_xaxes(title_text="Frequency [cm⁻¹]")
            fig_bar.update_layout(
                coloraxis_colorbar = dict(
                    title = "Local Mode",
                )
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            html = """
                    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>                     
                    <style>
                        .mol-container {
                        width: 100%;
                        height: 400px;
                        position: relative;
                        }
                    </style>
                    <div id="container-02" class="mol-container"></div>
                    <script>
                        let element = 'container-02';
                        let config = {};
                        let labelSpec = {
                            backgroundColor:"white",
                            fontColor:"black",
                            backgroundOpacity:0.75,
                            inFront:true,
                        };
                        let v = $3Dmol.createViewer( element, config );
                        var m = v.addModel(`"""+pdb.decode("utf-8")+"""`, "pdb", {keepH:true, assignBonds:true});
                        """+''.join(list(map(lambda i: f"""v.addLabel("{i[0]+1}", labelSpec, {{index: {i[0]}}});""", enumerate(tensor_props[index]["[atoms]"]))))+"""
                        v.setBackgroundColor(0xffffff, 0.0);
                        v.setStyle({},{cartoon:{}, sphere:{scale:0.25, colorscheme:'Jmol'}, stick:{radius:0.15, colorscheme:'Jmol'}});
                            v.vibrate(10, 1);
                            v.animate({loop: "forward",reps: 1});
                            v.zoomTo();                                      /* set camera */
                            v.render();                                      /* render scene */
                    </script>"""
            st.components.v1.html(html, height = 425)
        except:
            st.write("Due to current limitations, molecules with aromatic rings are not included in the local mode analysis.")
            # {y[1]: {x[0]: x[1] for x in zip(tensor_props[index]["local [modes]"], tensor_props[index]["local [contributions]"][y[0]])} for y in enumerate(tensor_props[index]["Frequency [cm⁻¹]"])}
            # st.write(sum(tensor_props[index]["local [contributions]"][1]))
            # tensor_props[index]["local [modes]"]
            # tensor_props[index]["Frequency [cm⁻¹]"]
with cc2:
    df = pd.DataFrame({"Property": mol_props.keys(), "Value": mol_props.values()})
    df.loc[-1] = ["CASRN", casrn]
    df.sort_values(by=['Property'], inplace=True)
    html = f"""
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>                     
        <style>
            .mol-container {{
            width: 100%;
            height: 500px;
            position: relative;
            }}
        </style>
        <center><div id="container-022" class="mol-container"></div></center>
        <script>
            let element = 'container-022';
            let config = {{}};
            let labelSpec = {{
                alignment:"center",
                backgroundColor:"white",
                fontColor:"black",
                backgroundOpacity:0.5,
                inFront:true,
            }};
            let v = $3Dmol.createViewer( element, config );
        v.addModel(`
HETATM    1  C01 UNK     1      16.649   6.683 -32.551  0.00  0.00           C  
HETATM    2  N01 UNK     1      17.870  10.541 -33.942  0.00  0.00           N  
HETATM    3  C02 UNK     1      17.692   6.796 -33.484  0.00  0.00           C  
HETATM    4  C03 UNK     1      18.093   8.062 -33.941  0.00  0.00           C  
HETATM    5  C04 UNK     1      17.451   9.216 -33.465  0.00  0.00           C  
HETATM    6  C05 UNK     1      16.406   9.105 -32.531  0.00  0.00           C  
HETATM    7  C06 UNK     1      16.005   7.838 -32.075  0.00  0.00           C  
HETATM    8  C07 UNK     1      15.702  10.370 -32.008  0.00  0.00           C  
HETATM    9  C08 UNK     1      16.209   5.294 -32.051  0.00  0.00           C  
HETATM   10  C09 UNK     1      17.415   4.336 -32.071  0.00  0.00           C  
HETATM   11  C10 UNK     1      15.673   5.413 -30.613  0.00  0.00           C  
HETATM   12  F01 UNK     1      14.331   5.559 -30.641  0.00  0.00           F  
HETATM   13  F02 UNK     1      16.230   6.485 -30.010  0.00  0.00           F  
HETATM   14  F03 UNK     1      15.989   4.296 -29.922  0.00  0.00           F  
HETATM   15  F04 UNK     1      15.238   4.812 -32.856  0.00  0.00           F  
HETATM   16  F05 UNK     1      16.997   3.094 -32.392  0.00  0.00           F  
HETATM   17  F06 UNK     1      18.311   4.759 -32.989  0.00  0.00           F  
HETATM   18  F07 UNK     1      14.419  10.077 -31.704  0.00  0.00           F  
HETATM   19  F08 UNK     1      16.337  10.817 -30.905  0.00  0.00           F  
HETATM   20  F09 UNK     1      15.729  11.325 -32.961  0.00  0.00           F  
HETATM   21  F10 UNK     1      17.994   4.317 -30.853  0.00  0.00           F  
HETATM   22  I01 UNK     1      19.683   8.234 -35.362  0.00  0.00           I  
HETATM   23  H01 UNK     1      17.407  11.371 -33.599  0.00  0.00           H  
HETATM   24  H02 UNK     1      18.619  10.622 -34.613  0.00  0.00           H  
HETATM   25  H08 UNK     1      18.198   5.888 -33.859  0.00  0.00           H  
HETATM   26  H12 UNK     1      15.186   7.749 -31.341  0.00  0.00           H  
HETATM   27  O01 UNK A   2      30.978   2.020 -23.792  0.00  0.00           O  
HETATM   28  H01 UNK A   2      30.356   1.540 -23.240  0.00  0.00           H  
HETATM   29  H02 UNK A   2      30.752   2.933 -23.983  0.00  0.00           H  
HETATM   30  O01 UNK A   4      32.365  13.413 -25.637  0.00  0.00           O  
HETATM   31  H01 UNK A   4      32.114  13.180 -26.535  0.00  0.00           H  
HETATM   32  H02 UNK A   4      31.842  13.014 -24.937  0.00  0.00           H  
HETATM   33  O01 UNK A   6      18.048   4.211 -16.510  0.00  0.00           O  
HETATM   34  H01 UNK A   6      17.188   4.184 -16.939  0.00  0.00           H  
HETATM   35  H02 UNK A   6      18.523   5.042 -16.575  0.00  0.00           H  
HETATM   36  O01 UNK A   7      30.014   8.200 -27.549  0.00  0.00           O  
HETATM   37  H01 UNK A   7      30.900   7.918 -27.793  0.00  0.00           H  
HETATM   38  H02 UNK A   7      29.333   7.524 -27.590  0.00  0.00           H  
HETATM   39  O01 UNK A   8      23.931  18.184 -22.984  0.00  0.00           O  
HETATM   40  H01 UNK A   8      24.284  17.641 -22.274  0.00  0.00           H  
HETATM   41  H02 UNK A   8      23.002  18.057 -23.188  0.00  0.00           H  
HETATM   42  O01 UNK A   9      27.418  17.936 -15.316  0.00  0.00           O  
HETATM   43  H01 UNK A   9      26.721  17.548 -15.852  0.00  0.00           H  
HETATM   44  H02 UNK A   9      28.269  17.491 -15.347  0.00  0.00           H  
HETATM   45  O01 UNK A  10      14.117  17.868 -21.601  0.00  0.00           O  
HETATM   46  H01 UNK A  10      13.394  17.235 -21.567  0.00  0.00           H  
HETATM   47  H02 UNK A  10      14.378  18.253 -20.762  0.00  0.00           H  
HETATM   48  O01 UNK A  11      18.367  11.854 -15.841  0.00  0.00           O  
HETATM   49  H01 UNK A  11      19.079  11.950 -16.479  0.00  0.00           H  
HETATM   50  H02 UNK A  11      17.638  12.472 -15.929  0.00  0.00           H  
HETATM   51  O01 UNK A  12      15.678   9.643 -20.218  0.00  0.00           O  
HETATM   52  H01 UNK A  12      14.869   9.341 -20.641  0.00  0.00           H  
HETATM   53  H02 UNK A  12      16.479   9.176 -20.464  0.00  0.00           H  
HETATM   54  O01 UNK A  13      28.825  18.168 -25.490  0.00  0.00           O  
HETATM   55  H01 UNK A  13      28.858  17.606 -26.268  0.00  0.00           H  
HETATM   56  H02 UNK A  13      29.557  18.079 -24.875  0.00  0.00           H  
HETATM   57  O01 UNK A  14      23.453   1.760 -19.978  0.00  0.00           O  
HETATM   58  H01 UNK A  14      22.950   1.774 -20.797  0.00  0.00           H  
HETATM   59  H02 UNK A  14      23.680   0.891 -19.639  0.00  0.00           H  
HETATM   60  O01 UNK A  15      20.304  17.054 -18.502  0.00  0.00           O  
HETATM   61  H01 UNK A  15      20.561  17.941 -18.237  0.00  0.00           H  
HETATM   62  H02 UNK A  15      19.391  16.946 -18.776  0.00  0.00           H  
HETATM   63  O01 UNK A  16      31.512   6.796 -19.118  0.00  0.00           O  
HETATM   64  H01 UNK A  16      31.980   7.035 -18.314  0.00  0.00           H  
HETATM   65  H02 UNK A  16      31.678   5.912 -19.455  0.00  0.00           H  
HETATM   66  O01 UNK A  17      12.180  15.630 -16.117  0.00  0.00           O  
HETATM   67  H01 UNK A  17      11.829  16.518 -16.207  0.00  0.00           H  
HETATM   68  H02 UNK A  17      11.890  14.994 -16.775  0.00  0.00           H  
HETATM   69  O01 UNK A  20      31.990   5.754 -35.349  0.00  0.00           O  
HETATM   70  H01 UNK A  20      31.315   5.297 -34.840  0.00  0.00           H  
HETATM   71  H02 UNK A  20      32.330   6.564 -34.963  0.00  0.00           H  
HETATM   72  N   UNL     1      57.369   3.750 -12.654  0.00  0.00           N  
HETATM   73  N   UNL     1      57.330   3.917  -2.611  0.00  0.00           N  
HETATM   74  N   UNL     1      50.220  11.003 -12.728  0.00  0.00           N  
HETATM   75  N   UNL     1      50.135  11.014  -2.619  0.00  0.00           N  
HETATM   76  N   UNL     1      43.041   3.930 -12.728  0.00  0.00           N  
HETATM   77  N   UNL     1      42.983   3.796  -2.694  0.00  0.00           N  
HETATM   78  N   UNL     1      49.960  -3.366 -12.595  0.00  0.00           N  
HETATM   79  N   UNL     1      50.308  -3.319  -2.663  0.00  0.00           N  
HETATM   80  C   UNL     1      53.918   3.817 -10.264  0.00  0.00           C  
HETATM   81  C   UNL     1      53.839   3.652 -11.646  0.00  0.00           C  
HETATM   82  C   UNL     1      54.966   3.653 -12.440  0.00  0.00           C  
HETATM   83  C   UNL     1      56.231   3.817 -11.873  0.00  0.00           C  
HETATM   84  C   UNL     1      56.322   3.979 -10.491  0.00  0.00           C  
HETATM   85  C   UNL     1      55.186   3.978  -9.709  0.00  0.00           C  
HETATM   86  C   UNL     1      53.901   3.914  -5.033  0.00  0.00           C  
HETATM   87  C   UNL     1      54.106   4.901  -4.071  0.00  0.00           C  
HETATM   88  C   UNL     1      55.241   4.921  -3.287  0.00  0.00           C  
HETATM   89  C   UNL     1      56.223   3.942  -3.438  0.00  0.00           C  
HETATM   90  C   UNL     1      56.029   2.949  -4.399  0.00  0.00           C  
HETATM   91  C   UNL     1      54.889   2.940  -5.175  0.00  0.00           C  
HETATM   92  C   UNL     1      50.248   7.589 -10.285  0.00  0.00           C  
HETATM   93  C   UNL     1      49.634   8.770  -9.871  0.00  0.00           C  
HETATM   94  C   UNL     1      49.642   9.903 -10.658  0.00  0.00           C  
HETATM   95  C   UNL     1      50.269   9.895 -11.904  0.00  0.00           C  
HETATM   96  C   UNL     1      50.891   8.719 -12.327  0.00  0.00           C  
HETATM   97  C   UNL     1      50.876   7.594 -11.529  0.00  0.00           C  
HETATM   98  C   UNL     1      50.126   7.594  -5.054  0.00  0.00           C  
HETATM   99  C   UNL     1      50.769   8.765  -5.453  0.00  0.00           C  
HETATM  100  C   UNL     1      50.756   9.900  -4.669  0.00  0.00           C  
HETATM  101  C   UNL     1      50.093   9.904  -3.441  0.00  0.00           C  
HETATM  102  C   UNL     1      49.444   8.738  -3.033  0.00  0.00           C  
HETATM  103  C   UNL     1      49.465   7.611  -3.828  0.00  0.00           C  
HETATM  104  C   UNL     1      46.469   3.918 -10.304  0.00  0.00           C  
HETATM  105  C   UNL     1      46.240   4.935 -11.229  0.00  0.00           C  
HETATM  106  C   UNL     1      45.105   4.957 -12.013  0.00  0.00           C  
HETATM  107  C   UNL     1      44.147   3.950 -11.899  0.00  0.00           C  
HETATM  108  C   UNL     1      44.365   2.926 -10.976  0.00  0.00           C  
HETATM  109  C   UNL     1      45.504   2.917 -10.200  0.00  0.00           C  
HETATM  110  C   UNL     1      46.443   3.841  -5.073  0.00  0.00           C  
HETATM  111  C   UNL     1      46.515   3.664  -3.692  0.00  0.00           C  
HETATM  112  C   UNL     1      45.385   3.671  -2.901  0.00  0.00           C  
HETATM  113  C   UNL     1      44.125   3.856  -3.471  0.00  0.00           C  
HETATM  114  C   UNL     1      44.040   4.032  -4.852  0.00  0.00           C  
HETATM  115  C   UNL     1      45.179   4.023  -5.630  0.00  0.00           C  
HETATM  116  C   UNL     1      50.109   0.120 -10.261  0.00  0.00           C  
HETATM  117  C   UNL     1      49.506   0.116 -11.517  0.00  0.00           C  
HETATM  118  C   UNL     1      49.474  -1.020 -12.300  0.00  0.00           C  
HETATM  119  C   UNL     1      50.048  -2.208 -11.846  0.00  0.00           C  
HETATM  120  C   UNL     1      50.648  -2.217 -10.587  0.00  0.00           C  
HETATM  121  C   UNL     1      50.674  -1.074  -9.816  0.00  0.00           C  
HETATM  122  C   UNL     1      50.248   0.132  -5.052  0.00  0.00           C  
HETATM  123  C   UNL     1      50.970   0.089  -3.860  0.00  0.00           C  
HETATM  124  C   UNL     1      51.013  -1.049  -3.082  0.00  0.00           C  
HETATM  125  C   UNL     1      50.325  -2.198  -3.471  0.00  0.00           C  
HETATM  126  C   UNL     1      49.601  -2.168  -4.664  0.00  0.00           C  
HETATM  127  C   UNL     1      49.567  -1.023  -5.432  0.00  0.00           C  
HETATM  128  O   UNL     1      52.838   3.851  -7.655  0.00  0.00           O  
HETATM  129  O   UNL     1      50.184   6.528  -7.669  0.00  0.00           O  
HETATM  130  O   UNL     1      47.529   3.870  -7.679  0.00  0.00           O  
HETATM  131  O   UNL     1      50.183   1.210  -7.659  0.00  0.00           O  
HETATM  132  O   UNL     1      51.531   5.177  -9.552  0.00  0.00           O  
HETATM  133  O   UNL     1      51.508   5.218  -5.778  0.00  0.00           O  
HETATM  134  O   UNL     1      48.871   5.210  -9.564  0.00  0.00           O  
HETATM  135  O   UNL     1      48.846   5.181  -5.779  0.00  0.00           O  
HETATM  136  O   UNL     1      48.841   2.543  -9.559  0.00  0.00           O  
HETATM  137  O   UNL     1      48.856   2.525  -5.777  0.00  0.00           O  
HETATM  138  O   UNL     1      51.499   2.518  -9.554  0.00  0.00           O  
HETATM  139  O   UNL     1      51.517   2.550  -5.762  0.00  0.00           O  
HETATM  140 SI   UNL     1      52.372   3.836  -9.204  0.00  0.00          Si  
HETATM  141 SI   UNL     1      52.365   3.885  -6.107  0.00  0.00          Si  
HETATM  142 SI   UNL     1      50.211   6.050  -9.215  0.00  0.00          Si  
HETATM  143 SI   UNL     1      50.164   6.054  -6.122  0.00  0.00          Si  
HETATM  144 SI   UNL     1      48.002   3.887  -9.227  0.00  0.00          Si  
HETATM  145 SI   UNL     1      47.991   3.848  -6.129  0.00  0.00          Si  
HETATM  146 SI   UNL     1      50.159   1.671  -9.209  0.00  0.00          Si  
HETATM  147 SI   UNL     1      50.198   1.678  -6.110  0.00  0.00          Si  
HETATM  148  H   UNL     1      58.201   4.158 -12.259  0.00  0.00           H  
HETATM  149  H   UNL     1      57.243   3.927 -13.638  0.00  0.00           H  
HETATM  150  H   UNL     1      52.869   3.521 -12.099  0.00  0.00           H  
HETATM  151  H   UNL     1      54.882   3.524 -13.511  0.00  0.00           H  
HETATM  152  H   UNL     1      57.296   4.106 -10.038  0.00  0.00           H  
HETATM  153  H   UNL     1      55.271   4.105  -8.641  0.00  0.00           H  
HETATM  154  H   UNL     1      58.123   3.393  -2.946  0.00  0.00           H  
HETATM  155  H   UNL     1      57.564   4.786  -2.158  0.00  0.00           H  
HETATM  156  H   UNL     1      53.357   5.667  -3.948  0.00  0.00           H  
HETATM  157  H   UNL     1      55.380   5.698  -2.548  0.00  0.00           H  
HETATM  158  H   UNL     1      56.784   2.184  -4.527  0.00  0.00           H  
HETATM  159  H   UNL     1      54.755   2.167  -5.916  0.00  0.00           H  
HETATM  160  H   UNL     1      50.916  11.047 -13.455  0.00  0.00           H  
HETATM  161  H   UNL     1      50.033  11.884 -12.275  0.00  0.00           H  
HETATM  162  H   UNL     1      49.151   8.793  -8.906  0.00  0.00           H  
HETATM  163  H   UNL     1      49.161  10.810 -10.316  0.00  0.00           H  
HETATM  164  H   UNL     1      51.386   8.702 -13.288  0.00  0.00           H  
HETATM  165  H   UNL     1      51.369   6.695 -11.864  0.00  0.00           H  
HETATM  166  H   UNL     1      49.419  11.068  -1.912  0.00  0.00           H  
HETATM  167  H   UNL     1      50.345  11.890  -3.068  0.00  0.00           H  
HETATM  168  H   UNL     1      51.279   8.779  -6.404  0.00  0.00           H  
HETATM  169  H   UNL     1      51.259  10.798  -4.999  0.00  0.00           H  
HETATM  170  H   UNL     1      48.922   8.730  -2.086  0.00  0.00           H  
HETATM  171  H   UNL     1      48.949   6.720  -3.505  0.00  0.00           H  
HETATM  172  H   UNL     1      42.262   3.373 -12.414  0.00  0.00           H  
HETATM  173  H   UNL     1      42.785   4.809 -13.147  0.00  0.00           H  
HETATM  174  H   UNL     1      46.972   5.722 -11.325  0.00  0.00           H  
HETATM  175  H   UNL     1      44.948   5.757 -12.724  0.00  0.00           H  
HETATM  176  H   UNL     1      43.629   2.140 -10.877  0.00  0.00           H  
HETATM  177  H   UNL     1      45.657   2.120  -9.488  0.00  0.00           H  
HETATM  178  H   UNL     1      42.157   4.216  -3.087  0.00  0.00           H  
HETATM  179  H   UNL     1      43.108   3.959  -1.708  0.00  0.00           H  
HETATM  180  H   UNL     1      47.482   3.516  -3.237  0.00  0.00           H  
HETATM  181  H   UNL     1      45.465   3.532  -1.832  0.00  0.00           H  
HETATM  182  H   UNL     1      43.069   4.174  -5.306  0.00  0.00           H  
HETATM  183  H   UNL     1      45.098   4.159  -6.698  0.00  0.00           H  
HETATM  184  H   UNL     1      50.624  -4.090 -12.372  0.00  0.00           H  
HETATM  185  H   UNL     1      49.791  -3.244 -13.581  0.00  0.00           H  
HETATM  186  H   UNL     1      49.052   1.026 -11.878  0.00  0.00           H  
HETATM  187  H   UNL     1      49.001  -1.000 -13.272  0.00  0.00           H  
HETATM  188  H   UNL     1      51.093  -3.133 -10.222  0.00  0.00           H  
HETATM  189  H   UNL     1      51.137  -1.097  -8.842  0.00  0.00           H  
HETATM  190  H   UNL     1      50.056  -4.185  -3.110  0.00  0.00           H  
HETATM  191  H   UNL     1      51.055  -3.394  -1.992  0.00  0.00           H  
HETATM  192  H   UNL     1      51.516   0.967  -3.552  0.00  0.00           H  
HETATM  193  H   UNL     1      51.581  -1.061  -2.162  0.00  0.00           H  
HETATM  194  H   UNL     1      49.068  -3.054  -4.979  0.00  0.00           H  
HETATM  195  H   UNL     1      49.011  -1.018  -6.356  0.00  0.00           H  
HETATM  196  O   UNK     1      11.669  32.742 -23.753  1.00  0.00           O  
HETATM  197  H   UNK     2      11.446  32.092 -23.083  1.00  0.00           H  
HETATM  198  O   UNK     3      11.669  35.014 -22.466  1.00  0.00           O  
HETATM  199  H   UNK     4      11.523  35.711 -23.109  1.00  0.00           H  
HETATM  200 Si   UNK     5      12.601  33.878 -23.110  1.00  0.00          Si  
HETATM  201  O   UNK     6      13.046  31.381 -25.555  1.00  0.00           O  
HETATM  202  H   UNK     7      12.910  32.090 -24.922  1.00  0.00           H  
HETATM  203  O   UNK     8      13.046  31.381 -20.150  1.00  0.00           O  
HETATM  204  H   UNK     9      12.910  32.090 -19.517  1.00  0.00           H  
HETATM  205  O   UNK    10      13.046  36.375 -26.070  1.00  0.00           O  
HETATM  206  H   UNK    11      12.778  35.624 -26.605  1.00  0.00           H  
HETATM  207  O   UNK    12      13.046  36.375 -20.664  1.00  0.00           O  
HETATM  208  H   UNK    13      12.778  35.624 -21.200  1.00  0.00           H  
HETATM  209  O   UNK    14      13.536  33.254 -27.357  1.00  0.00           O  
HETATM  210  H   UNK    15      13.370  33.127 -28.294  1.00  0.00           H  
HETATM  211  O   UNK    16      13.536  33.254 -21.952  1.00  0.00           O  
HETATM  212  O   UNK    17      13.536  34.502 -24.268  1.00  0.00           O  
HETATM  213  O   UNK    18      13.536  34.502 -18.862  1.00  0.00           O  
HETATM  214  H   UNK    19      13.489  34.592 -17.908  1.00  0.00           H  
HETATM  215 Si   UNK    20      14.054  31.878 -26.713  1.00  0.00          Si  
HETATM  216 Si   UNK    21      14.054  31.878 -21.308  1.00  0.00          Si  
HETATM  217 Si   UNK    22      14.054  35.877 -24.912  1.00  0.00          Si  
HETATM  218 Si   UNK    23      14.054  35.877 -19.506  1.00  0.00          Si  
HETATM  219  O   UNK    24      14.127  28.484 -23.753  1.00  0.00           O  
HETATM  220  H   UNK    25      13.904  27.834 -23.083  1.00  0.00           H  
HETATM  221  O   UNK    26      14.127  28.484 -18.348  1.00  0.00           O  
HETATM  222  H   UNK    27      13.904  27.834 -17.678  1.00  0.00           H  
HETATM  223  O   UNK    28      14.127  30.757 -27.871  1.00  0.00           O  
HETATM  224  H   UNK    29      13.592  31.033 -28.619  1.00  0.00           H  
HETATM  225  O   UNK    30      14.127  30.757 -22.466  1.00  0.00           O  
HETATM  226  O   UNK    31      14.127  30.757 -17.061  1.00  0.00           O  
HETATM  227  H   UNK    32      13.981  31.454 -17.704  1.00  0.00           H  
HETATM  228  O   UNK    33      14.127  36.999 -29.159  1.00  0.00           O  
HETATM  229  H   UNK    34      13.904  36.349 -28.488  1.00  0.00           H  
HETATM  230  O   UNK    35      14.127  36.999 -23.753  1.00  0.00           O  
HETATM  231  O   UNK    36      14.127  36.999 -18.348  1.00  0.00           O  
HETATM  232  H   UNK    37      14.715  36.701 -17.650  1.00  0.00           H  
HETATM  233  O   UNK    38      14.127  39.271 -27.871  1.00  0.00           O  
HETATM  234  H   UNK    39      13.981  39.969 -28.515  1.00  0.00           H  
HETATM  235  O   UNK    40      14.127  39.271 -22.466  1.00  0.00           O  
HETATM  236  H   UNK    41      13.981  39.969 -23.109  1.00  0.00           H  
HETATM  237 Si   UNK    42      15.059  29.620 -23.110  1.00  0.00          Si  
HETATM  238 Si   UNK    43      15.059  29.620 -17.704  1.00  0.00          Si  
HETATM  239 Si   UNK    44      15.059  38.135 -28.515  1.00  0.00          Si  
HETATM  240 Si   UNK    45      15.059  38.135 -23.110  1.00  0.00          Si  
HETATM  241  O   UNK    46      15.504  27.123 -25.555  1.00  0.00           O  
HETATM  242  H   UNK    47      15.368  27.832 -24.922  1.00  0.00           H  
HETATM  243  O   UNK    48      15.504  27.123 -20.150  1.00  0.00           O  
HETATM  244  H   UNK    49      15.368  27.832 -19.517  1.00  0.00           H  
HETATM  245  O   UNK    50      15.504  32.117 -31.475  1.00  0.00           O  
HETATM  246  H   UNK    51      15.236  31.367 -32.010  1.00  0.00           H  
HETATM  247  O   UNK    52      15.504  32.117 -26.070  1.00  0.00           O  
HETATM  248  O   UNK    53      15.504  32.117 -20.664  1.00  0.00           O  
HETATM  249  O   UNK    54      15.504  35.638 -25.555  1.00  0.00           O  
HETATM  250  O   UNK    55      15.504  35.638 -20.150  1.00  0.00           O  
HETATM  251  O   UNK    56      15.504  35.638 -14.745  1.00  0.00           O  
HETATM  252  H   UNK    57      15.368  36.347 -14.112  1.00  0.00           H  
HETATM  253  O   UNK    58      15.504  40.632 -26.070  1.00  0.00           O  
HETATM  254  H   UNK    59      15.236  39.882 -26.605  1.00  0.00           H  
HETATM  255  O   UNK    60      15.504  40.632 -20.664  1.00  0.00           O  
HETATM  256  H   UNK    61      15.236  39.882 -21.200  1.00  0.00           H  
HETATM  257  O   UNK    62      15.994  28.996 -27.357  1.00  0.00           O  
HETATM  258  H   UNK    63      15.828  28.869 -28.294  1.00  0.00           H  
HETATM  259  O   UNK    64      15.994  28.996 -21.952  1.00  0.00           O  
HETATM  260  O   UNK    65      15.994  28.996 -16.546  1.00  0.00           O  
HETATM  261  H   UNK    66      16.246  28.101 -16.786  1.00  0.00           H  
HETATM  262  O   UNK    67      15.994  30.245 -29.673  1.00  0.00           O  
HETATM  263  H   UNK    68      15.947  30.335 -28.719  1.00  0.00           H  
HETATM  264  O   UNK    69      15.994  30.245 -24.268  1.00  0.00           O  
HETATM  265  O   UNK    70      15.994  30.245 -18.862  1.00  0.00           O  
HETATM  266  O   UNK    71      15.994  37.511 -27.357  1.00  0.00           O  
HETATM  267  O   UNK    72      15.994  37.511 -21.952  1.00  0.00           O  
HETATM  268  O   UNK    73      15.994  37.511 -16.546  1.00  0.00           O  
HETATM  269  H   UNK    74      15.828  37.384 -17.483  1.00  0.00           H  
HETATM  270  O   UNK    75      15.994  38.759 -29.673  1.00  0.00           O  
HETATM  271  H   UNK    76      16.109  39.700 -29.520  1.00  0.00           H  
HETATM  272  O   UNK    77      15.994  38.759 -24.268  1.00  0.00           O  
HETATM  273  O   UNK    78      15.994  38.759 -18.862  1.00  0.00           O  
HETATM  274  H   UNK    79      15.947  38.850 -17.908  1.00  0.00           H  
HETATM  275 Si   UNK    80      16.512  27.621 -26.713  1.00  0.00          Si  
HETATM  276 Si   UNK    81      16.512  27.621 -21.308  1.00  0.00          Si  
HETATM  277 Si   UNK    82      16.512  31.620 -30.317  1.00  0.00          Si  
HETATM  278 Si   UNK    83      16.512  31.620 -24.912  1.00  0.00          Si  
HETATM  279 Si   UNK    84      16.512  31.620 -19.506  1.00  0.00          Si  
HETATM  280 Si   UNK    85      16.512  36.135 -26.713  1.00  0.00          Si  
HETATM  281 Si   UNK    86      16.512  36.135 -21.308  1.00  0.00          Si  
HETATM  282 Si   UNK    87      16.512  36.135 -15.903  1.00  0.00          Si  
HETATM  283 Si   UNK    88      16.512  40.135 -24.912  1.00  0.00          Si  
HETATM  284 Si   UNK    89      16.512  40.135 -19.506  1.00  0.00          Si  
HETATM  285  O   UNK    90      16.585  26.499 -27.871  1.00  0.00           O  
HETATM  286  H   UNK    91      16.050  26.776 -28.619  1.00  0.00           H  
HETATM  287  O   UNK    92      16.585  26.499 -22.466  1.00  0.00           O  
HETATM  288  H   UNK    93      16.050  26.776 -23.214  1.00  0.00           H  
HETATM  289  O   UNK    94      16.585  32.742 -29.159  1.00  0.00           O  
HETATM  290  O   UNK    95      16.585  32.742 -23.753  1.00  0.00           O  
HETATM  291  O   UNK    96      16.585  32.742 -18.348  1.00  0.00           O  
HETATM  292  O   UNK    97      16.585  35.014 -27.871  1.00  0.00           O  
HETATM  293  O   UNK    98      16.585  35.014 -22.466  1.00  0.00           O  
HETATM  294  O   UNK    99      16.585  35.014 -17.061  1.00  0.00           O  
HETATM  295  O   UNK   100      16.585  41.256 -23.753  1.00  0.00           O  
HETATM  296  H   UNK   101      17.173  40.958 -23.055  1.00  0.00           H  
HETATM  297  O   UNK   102      16.585  41.256 -18.348  1.00  0.00           O  
HETATM  298  H   UNK   103      17.173  40.958 -17.650  1.00  0.00           H  
HETATM  299 Si   UNK   104      17.517  33.878 -28.515  1.00  0.00          Si  
HETATM  300 Si   UNK   105      17.517  33.878 -23.110  1.00  0.00          Si  
HETATM  301 Si   UNK   106      17.517  33.878 -17.704  1.00  0.00          Si  
HETATM  302  O   UNK   107      17.962  27.860 -26.070  1.00  0.00           O  
HETATM  303  O   UNK   108      17.962  27.860 -20.664  1.00  0.00           O  
HETATM  304  O   UNK   109      17.962  31.381 -30.961  1.00  0.00           O  
HETATM  305  H   UNK   110      18.207  32.143 -31.491  1.00  0.00           H  
HETATM  306  O   UNK   111      17.962  31.381 -25.555  1.00  0.00           O  
HETATM  307  O   UNK   112      17.962  31.381 -20.150  1.00  0.00           O  
HETATM  308  O   UNK   113      17.962  31.381 -14.745  1.00  0.00           O  
HETATM  309  H   UNK   114      17.826  32.090 -14.112  1.00  0.00           H  
HETATM  310  O   UNK   115      17.962  36.375 -31.475  1.00  0.00           O  
HETATM  311  H   UNK   116      17.694  35.624 -32.010  1.00  0.00           H  
HETATM  312  O   UNK   117      17.962  36.375 -26.070  1.00  0.00           O  
HETATM  313  O   UNK   118      17.962  36.375 -20.664  1.00  0.00           O  
HETATM  314  O   UNK   119      17.962  36.375 -15.259  1.00  0.00           O  
HETATM  315  H   UNK   120      18.300  35.545 -14.914  1.00  0.00           H  
HETATM  316  O   UNK   121      17.962  39.896 -25.555  1.00  0.00           O  
HETATM  317  O   UNK   122      17.962  39.896 -20.150  1.00  0.00           O  
HETATM  318  O   UNK   123      18.452  25.987 -24.268  1.00  0.00           O  
HETATM  319  H   UNK   124      18.405  26.078 -23.313  1.00  0.00           H  
HETATM  320  O   UNK   125      18.452  25.987 -18.862  1.00  0.00           O  
HETATM  321  H   UNK   126      18.405  26.078 -17.908  1.00  0.00           H  
HETATM  322  O   UNK   127      18.452  33.254 -27.357  1.00  0.00           O  
HETATM  323  O   UNK   128      18.452  33.254 -21.952  1.00  0.00           O  
HETATM  324  O   UNK   129      18.452  33.254 -16.546  1.00  0.00           O  
HETATM  325  O   UNK   130      18.452  34.502 -29.673  1.00  0.00           O  
HETATM  326  O   UNK   131      18.452  34.502 -24.268  1.00  0.00           O  
HETATM  327  O   UNK   132      18.452  34.502 -18.862  1.00  0.00           O  
HETATM  328  O   UNK   133      18.452  41.768 -27.357  1.00  0.00           O  
HETATM  329  H   UNK   134      18.286  41.641 -28.294  1.00  0.00           H  
HETATM  330  O   UNK   135      18.452  41.768 -21.952  1.00  0.00           O  
HETATM  331  H   UNK   136      18.286  41.641 -22.889  1.00  0.00           H  
HETATM  332 Si   UNK   137      18.970  27.363 -24.912  1.00  0.00          Si  
HETATM  333 Si   UNK   138      18.970  27.363 -19.506  1.00  0.00          Si  
HETATM  334 Si   UNK   139      18.970  31.878 -26.713  1.00  0.00          Si  
HETATM  335 Si   UNK   140      18.970  31.878 -21.308  1.00  0.00          Si  
HETATM  336 Si   UNK   141      18.970  31.878 -15.903  1.00  0.00          Si  
HETATM  337 Si   UNK   142      18.970  35.877 -30.317  1.00  0.00          Si  
HETATM  338 Si   UNK   143      18.970  35.877 -24.912  1.00  0.00          Si  
HETATM  339 Si   UNK   144      18.970  35.877 -19.506  1.00  0.00          Si  
HETATM  340 Si   UNK   145      18.970  40.393 -26.713  1.00  0.00          Si  
HETATM  341 Si   UNK   146      18.970  40.393 -21.308  1.00  0.00          Si  
HETATM  342  O   UNK   147      19.043  28.484 -29.159  1.00  0.00           O  
HETATM  343  H   UNK   148      18.820  27.834 -28.488  1.00  0.00           H  
HETATM  344  O   UNK   149      19.043  28.484 -23.753  1.00  0.00           O  
HETATM  345  O   UNK   150      19.043  28.484 -18.348  1.00  0.00           O  
HETATM  346  O   UNK   151      19.043  30.757 -27.871  1.00  0.00           O  
HETATM  347  O   UNK   152      19.043  30.757 -22.466  1.00  0.00           O  
HETATM  348  O   UNK   153      19.043  30.757 -17.061  1.00  0.00           O  
HETATM  349  O   UNK   154      19.043  36.999 -29.159  1.00  0.00           O  
HETATM  350  O   UNK   155      19.043  36.999 -23.753  1.00  0.00           O  
HETATM  351  O   UNK   156      19.043  36.999 -18.348  1.00  0.00           O  
HETATM  352  O   UNK   157      19.043  39.271 -27.871  1.00  0.00           O  
HETATM  353  O   UNK   158      19.043  39.271 -22.466  1.00  0.00           O  
HETATM  354  O   UNK   159      19.043  39.271 -17.061  1.00  0.00           O  
HETATM  355  H   UNK   160      18.897  39.969 -17.704  1.00  0.00           H  
HETATM  356 Si   UNK   161      19.975  29.620 -28.515  1.00  0.00          Si  
HETATM  357 Si   UNK   162      19.975  29.620 -23.110  1.00  0.00          Si  
HETATM  358 Si   UNK   163      19.975  29.620 -17.704  1.00  0.00          Si  
HETATM  359 Si   UNK   164      19.975  38.135 -28.515  1.00  0.00          Si  
HETATM  360 Si   UNK   165      19.975  38.135 -23.110  1.00  0.00          Si  
HETATM  361 Si   UNK   166      19.975  38.135 -17.704  1.00  0.00          Si  
HETATM  362  O   UNK   167      20.420  27.123 -25.555  1.00  0.00           O  
HETATM  363  O   UNK   168      20.420  27.123 -20.150  1.00  0.00           O  
HETATM  364  O   UNK   169      20.420  32.117 -31.475  1.00  0.00           O  
HETATM  365  H   UNK   170      20.152  31.367 -32.010  1.00  0.00           H  
HETATM  366  O   UNK   171      20.420  32.117 -26.070  1.00  0.00           O  
HETATM  367  O   UNK   172      20.420  32.117 -20.664  1.00  0.00           O  
HETATM  368  O   UNK   173      20.420  32.117 -15.259  1.00  0.00           O  
HETATM  369  H   UNK   174      20.758  31.288 -14.914  1.00  0.00           H  
HETATM  370  O   UNK   175      20.420  35.638 -30.961  1.00  0.00           O  
HETATM  371  H   UNK   176      20.665  36.400 -31.491  1.00  0.00           H  
HETATM  372  O   UNK   177      20.420  35.638 -25.555  1.00  0.00           O  
HETATM  373  O   UNK   178      20.420  35.638 -20.150  1.00  0.00           O  
HETATM  374  O   UNK   179      20.420  35.638 -14.745  1.00  0.00           O  
HETATM  375  H   UNK   180      20.284  36.347 -14.112  1.00  0.00           H  
HETATM  376  O   UNK   181      20.420  40.632 -26.070  1.00  0.00           O  
HETATM  377  O   UNK   182      20.420  40.632 -20.664  1.00  0.00           O  
HETATM  378  O   UNK   183      20.910  28.996 -27.357  1.00  0.00           O  
HETATM  379  O   UNK   184      20.910  28.996 -21.952  1.00  0.00           O  
HETATM  380  O   UNK   185      20.910  28.996 -16.546  1.00  0.00           O  
HETATM  381  H   UNK   186      21.162  28.101 -16.786  1.00  0.00           H  
HETATM  382  O   UNK   187      20.910  30.245 -29.673  1.00  0.00           O  
HETATM  383  O   UNK   188      20.910  30.245 -24.268  1.00  0.00           O  
HETATM  384  O   UNK   189      20.910  30.245 -18.862  1.00  0.00           O  
HETATM  385  O   UNK   190      20.910  37.511 -27.357  1.00  0.00           O  
HETATM  386  O   UNK   191      20.910  37.511 -21.952  1.00  0.00           O  
HETATM  387  O   UNK   192      20.910  37.511 -16.546  1.00  0.00           O  
HETATM  388  O   UNK   193      20.910  38.759 -29.673  1.00  0.00           O  
HETATM  389  H   UNK   194      21.025  39.700 -29.520  1.00  0.00           H  
HETATM  390  O   UNK   195      20.910  38.759 -24.268  1.00  0.00           O  
HETATM  391  O   UNK   196      20.910  38.759 -18.862  1.00  0.00           O  
HETATM  392 Si   UNK   197      21.428  27.621 -26.713  1.00  0.00          Si  
HETATM  393 Si   UNK   198      21.428  27.621 -21.308  1.00  0.00          Si  
HETATM  394 Si   UNK   199      21.428  31.620 -30.317  1.00  0.00          Si  
HETATM  395 Si   UNK   200      21.428  31.620 -24.912  1.00  0.00          Si  
HETATM  396 Si   UNK   201      21.428  31.620 -19.506  1.00  0.00          Si  
HETATM  397 Si   UNK   202      21.428  36.135 -26.713  1.00  0.00          Si  
HETATM  398 Si   UNK   203      21.428  36.135 -21.308  1.00  0.00          Si  
HETATM  399 Si   UNK   204      21.428  36.135 -15.903  1.00  0.00          Si  
HETATM  400 Si   UNK   205      21.428  40.135 -24.912  1.00  0.00          Si  
HETATM  401 Si   UNK   206      21.428  40.135 -19.506  1.00  0.00          Si  
HETATM  402  O   UNK   207      21.501  24.227 -23.753  1.00  0.00           O  
HETATM  403  H   UNK   208      21.278  23.577 -23.083  1.00  0.00           H  
HETATM  404  O   UNK   209      21.501  26.499 -27.871  1.00  0.00           O  
HETATM  405  H   UNK   210      20.966  26.776 -28.619  1.00  0.00           H  
HETATM  406  O   UNK   211      21.501  26.499 -22.466  1.00  0.00           O  
HETATM  407  O   UNK   212      21.501  32.742 -29.159  1.00  0.00           O  
HETATM  408  O   UNK   213      21.501  32.742 -23.753  1.00  0.00           O  
HETATM  409  O   UNK   214      21.501  32.742 -18.348  1.00  0.00           O  
HETATM  410  O   UNK   215      21.501  35.014 -27.871  1.00  0.00           O  
HETATM  411  O   UNK   216      21.501  35.014 -22.466  1.00  0.00           O  
HETATM  412  O   UNK   217      21.501  35.014 -17.061  1.00  0.00           O  
HETATM  413  O   UNK   218      21.501  41.256 -23.753  1.00  0.00           O  
HETATM  414  O   UNK   219      21.501  41.256 -18.348  1.00  0.00           O  
HETATM  415  H   UNK   220      22.089  40.958 -17.650  1.00  0.00           H  
HETATM  416  O   UNK   221      21.501  43.529 -22.466  1.00  0.00           O  
HETATM  417  H   UNK   222      21.355  44.226 -23.109  1.00  0.00           H  
HETATM  418 Si   UNK   223      22.433  25.363 -23.110  1.00  0.00          Si  
HETATM  419 Si   UNK   224      22.433  33.878 -28.515  1.00  0.00          Si  
HETATM  420 Si   UNK   225      22.433  33.878 -23.110  1.00  0.00          Si  
HETATM  421 Si   UNK   226      22.433  33.878 -17.704  1.00  0.00          Si  
HETATM  422 Si   UNK   227      22.433  42.392 -23.110  1.00  0.00          Si  
HETATM  423  O   UNK   228      22.878  27.860 -26.070  1.00  0.00           O  
HETATM  424  O   UNK   229      22.878  27.860 -20.664  1.00  0.00           O  
HETATM  425  O   UNK   230      22.878  31.381 -30.961  1.00  0.00           O  
HETATM  426  H   UNK   231      23.123  32.143 -31.491  1.00  0.00           H  
HETATM  427  O   UNK   232      22.878  31.381 -25.555  1.00  0.00           O  
HETATM  428  O   UNK   233      22.878  31.381 -20.150  1.00  0.00           O  
HETATM  429  O   UNK   234      22.878  31.381 -14.745  1.00  0.00           O  
HETATM  430  H   UNK   235      22.742  32.090 -14.112  1.00  0.00           H  
HETATM  431  O   UNK   236      22.878  36.375 -31.475  1.00  0.00           O  
HETATM  432  H   UNK   237      22.610  35.624 -32.010  1.00  0.00           H  
HETATM  433  O   UNK   238      22.878  36.375 -26.070  1.00  0.00           O  
HETATM  434  O   UNK   239      22.878  36.375 -20.664  1.00  0.00           O  
HETATM  435  O   UNK   240      22.878  36.375 -15.259  1.00  0.00           O  
HETATM  436  H   UNK   241      23.216  35.545 -14.913  1.00  0.00           H  
HETATM  437  O   UNK   242      22.878  39.896 -25.555  1.00  0.00           O  
HETATM  438  O   UNK   243      22.878  39.896 -20.150  1.00  0.00           O  
HETATM  439  O   UNK   244      23.368  24.739 -21.952  1.00  0.00           O  
HETATM  440  H   UNK   245      23.620  23.844 -22.191  1.00  0.00           H  
HETATM  441  O   UNK   246      23.368  25.987 -24.268  1.00  0.00           O  
HETATM  442  O   UNK   247      23.368  25.987 -18.862  1.00  0.00           O  
HETATM  443  H   UNK   248      23.321  26.078 -17.908  1.00  0.00           H  
HETATM  444  O   UNK   249      23.368  33.254 -27.357  1.00  0.00           O  
HETATM  445  O   UNK   250      23.368  33.254 -21.952  1.00  0.00           O  
HETATM  446  O   UNK   251      23.368  33.254 -16.546  1.00  0.00           O  
HETATM  447  O   UNK   252      23.368  34.502 -29.673  1.00  0.00           O  
HETATM  448  O   UNK   253      23.368  34.502 -24.268  1.00  0.00           O  
HETATM  449  O   UNK   254      23.368  34.502 -18.862  1.00  0.00           O  
HETATM  450  O   UNK   255      23.368  41.768 -27.357  1.00  0.00           O  
HETATM  451  H   UNK   256      23.202  41.641 -28.294  1.00  0.00           H  
HETATM  452  O   UNK   257      23.368  41.768 -21.952  1.00  0.00           O  
HETATM  453  O   UNK   258      23.368  43.017 -24.268  1.00  0.00           O  
HETATM  454  H   UNK   259      23.483  43.957 -24.115  1.00  0.00           H  
HETATM  455 Si   UNK   260      23.886  27.363 -24.912  1.00  0.00          Si  
HETATM  456 Si   UNK   261      23.886  27.363 -19.506  1.00  0.00          Si  
HETATM  457 Si   UNK   262      23.886  31.878 -26.713  1.00  0.00          Si  
HETATM  458 Si   UNK   263      23.886  31.878 -21.308  1.00  0.00          Si  
HETATM  459 Si   UNK   264      23.886  31.878 -15.903  1.00  0.00          Si  
HETATM  460 Si   UNK   265      23.886  35.877 -30.317  1.00  0.00          Si  
HETATM  461 Si   UNK   266      23.886  35.877 -24.912  1.00  0.00          Si  
HETATM  462 Si   UNK   267      23.886  35.877 -19.506  1.00  0.00          Si  
HETATM  463 Si   UNK   268      23.886  40.393 -26.713  1.00  0.00          Si  
HETATM  464 Si   UNK   269      23.886  40.393 -21.308  1.00  0.00          Si  
HETATM  465  O   UNK   270      23.959  28.484 -29.159  1.00  0.00           O  
HETATM  466  H   UNK   271      23.736  27.834 -28.488  1.00  0.00           H  
HETATM  467  O   UNK   272      23.959  28.484 -23.753  1.00  0.00           O  
HETATM  468  O   UNK   273      23.959  28.484 -18.348  1.00  0.00           O  
HETATM  469  O   UNK   274      23.959  30.757 -27.871  1.00  0.00           O  
HETATM  470  O   UNK   275      23.959  30.757 -22.466  1.00  0.00           O  
HETATM  471  O   UNK   276      23.959  30.757 -17.061  1.00  0.00           O  
HETATM  472  O   UNK   277      23.959  36.999 -29.159  1.00  0.00           O  
HETATM  473  O   UNK   278      23.959  36.999 -23.753  1.00  0.00           O  
HETATM  474  O   UNK   279      23.959  36.999 -18.348  1.00  0.00           O  
HETATM  475  O   UNK   280      23.959  39.271 -27.871  1.00  0.00           O  
HETATM  476  O   UNK   281      23.959  39.271 -22.466  1.00  0.00           O  
HETATM  477  O   UNK   282      23.959  39.271 -17.061  1.00  0.00           O  
HETATM  478  H   UNK   283      23.813  39.969 -17.704  1.00  0.00           H  
HETATM  479 Si   UNK   284      24.891  29.620 -28.515  1.00  0.00          Si  
HETATM  480 Si   UNK   285      24.891  29.620 -23.110  1.00  0.00          Si  
HETATM  481 Si   UNK   286      24.891  29.620 -17.704  1.00  0.00          Si  
HETATM  482 Si   UNK   287      24.891  38.135 -28.515  1.00  0.00          Si  
HETATM  483 Si   UNK   288      24.891  38.135 -23.110  1.00  0.00          Si  
HETATM  484 Si   UNK   289      24.891  38.135 -17.704  1.00  0.00          Si  
HETATM  485  O   UNK   290      25.336  27.123 -25.555  1.00  0.00           O  
HETATM  486  H   UNK   291      25.581  27.885 -26.086  1.00  0.00           H  
HETATM  487  O   UNK   292      25.336  27.123 -20.150  1.00  0.00           O  
HETATM  488  H   UNK   293      25.581  27.885 -20.680  1.00  0.00           H  
HETATM  489  O   UNK   294      25.336  32.117 -26.070  1.00  0.00           O  
HETATM  490  O   UNK   295      25.336  32.117 -20.664  1.00  0.00           O  
HETATM  491  O   UNK   296      25.336  32.117 -15.259  1.00  0.00           O  
HETATM  492  H   UNK   297      25.674  31.288 -14.914  1.00  0.00           H  
HETATM  493  O   UNK   298      25.336  35.638 -30.961  1.00  0.00           O  
HETATM  494  H   UNK   299      25.581  36.400 -31.491  1.00  0.00           H  
HETATM  495  O   UNK   300      25.336  35.638 -25.555  1.00  0.00           O  
HETATM  496  O   UNK   301      25.336  35.638 -20.150  1.00  0.00           O  
HETATM  497  O   UNK   302      25.336  40.632 -26.070  1.00  0.00           O  
HETATM  498  H   UNK   303      25.674  39.803 -25.724  1.00  0.00           H  
HETATM  499  O   UNK   304      25.336  40.632 -20.664  1.00  0.00           O  
HETATM  500  H   UNK   305      25.674  39.803 -20.319  1.00  0.00           H  
HETATM  501  O   UNK   306      25.826  28.996 -27.357  1.00  0.00           O  
HETATM  502  H   UNK   307      26.078  28.101 -27.596  1.00  0.00           H  
HETATM  503  O   UNK   308      25.826  28.996 -21.952  1.00  0.00           O  
HETATM  504  H   UNK   309      26.078  28.101 -22.191  1.00  0.00           H  
HETATM  505  O   UNK   310      25.826  28.996 -16.546  1.00  0.00           O  
HETATM  506  H   UNK   311      26.078  28.101 -16.786  1.00  0.00           H  
HETATM  507  O   UNK   312      25.826  30.245 -29.673  1.00  0.00           O  
HETATM  508  H   UNK   313      25.941  31.185 -29.520  1.00  0.00           H  
HETATM  509  O   UNK   314      25.826  30.245 -24.268  1.00  0.00           O  
HETATM  510  O   UNK   315      25.826  30.245 -18.862  1.00  0.00           O  
HETATM  511  O   UNK   316      25.826  37.511 -27.357  1.00  0.00           O  
HETATM  512  O   UNK   317      25.826  37.511 -21.952  1.00  0.00           O  
HETATM  513  O   UNK   318      25.826  37.511 -16.546  1.00  0.00           O  
HETATM  514  H   UNK   319      26.078  36.616 -16.786  1.00  0.00           H  
HETATM  515  O   UNK   320      25.826  38.759 -29.673  1.00  0.00           O  
HETATM  516  H   UNK   321      25.941  39.700 -29.520  1.00  0.00           H  
HETATM  517  O   UNK   322      25.826  38.759 -24.268  1.00  0.00           O  
HETATM  518  H   UNK   323      25.941  39.700 -24.115  1.00  0.00           H  
HETATM  519  O   UNK   324      25.826  38.759 -18.862  1.00  0.00           O  
HETATM  520  H   UNK   325      25.941  39.700 -18.709  1.00  0.00           H  
HETATM  521 Si   UNK   326      26.344  31.620 -24.912  1.00  0.00          Si  
HETATM  522 Si   UNK   327      26.344  31.620 -19.506  1.00  0.00          Si  
HETATM  523 Si   UNK   328      26.344  36.135 -26.713  1.00  0.00          Si  
HETATM  524 Si   UNK   329      26.344  36.135 -21.308  1.00  0.00          Si  
HETATM  525  O   UNK   330      26.417  32.742 -23.753  1.00  0.00           O  
HETATM  526  O   UNK   331      26.417  32.742 -18.348  1.00  0.00           O  
HETATM  527  H   UNK   332      27.005  32.444 -17.650  1.00  0.00           H  
HETATM  528  O   UNK   333      26.417  35.014 -27.871  1.00  0.00           O  
HETATM  529  H   UNK   334      25.882  35.290 -28.619  1.00  0.00           H  
HETATM  530  O   UNK   335      26.417  35.014 -22.466  1.00  0.00           O  
HETATM  531 Si   UNK   336      27.349  33.878 -23.110  1.00  0.00          Si  
HETATM  532  O   UNK   337      27.794  31.381 -25.555  1.00  0.00           O  
HETATM  533  H   UNK   338      28.039  32.143 -26.086  1.00  0.00           H  
HETATM  534  O   UNK   339      27.794  31.381 -20.150  1.00  0.00           O  
HETATM  535  H   UNK   340      28.039  32.143 -20.680  1.00  0.00           H  
HETATM  536  O   UNK   341      27.794  36.375 -26.070  1.00  0.00           O  
HETATM  537  H   UNK   342      28.132  35.545 -25.724  1.00  0.00           H  
HETATM  538  O   UNK   343      27.794  36.375 -20.664  1.00  0.00           O  
HETATM  539  H   UNK   344      28.132  35.545 -20.319  1.00  0.00           H  
HETATM  540  O   UNK   345      28.284  33.254 -21.952  1.00  0.00           O  
HETATM  541  H   UNK   346      28.536  32.359 -22.191  1.00  0.00           H  
HETATM  542  O   UNK   347      28.284  34.502 -24.268  1.00  0.00           O  
HETATM  543  H   UNK   348      28.399  35.443 -24.115  1.00  0.00           H  
ATOM    544  O   ROOT    1      -8.052  15.956   4.084  0.00  0.00           O  
ATOM    545  O   ROOT    1      -3.921  16.193   7.537  0.00  0.00           O  
ATOM    546  O   ROOT    1      -4.612  11.824   4.458  0.00  0.00           O  
ATOM    547  O   ROOT    1      -0.482  12.061   7.911  0.00  0.00           O  
ATOM    548  O   ROOT    1      -1.173   7.692   4.833  0.00  0.00           O  
ATOM    549  O   ROOT    1       2.958   7.929   8.285  0.00  0.00           O  
ATOM    550  O   ROOT    1       2.266   3.560   5.207  0.00  0.00           O  
ATOM    551  O   ROOT    1       6.397   3.797   8.659  0.00  0.00           O  
ATOM    552  O   ROOT    1       5.705  -0.571   5.581  0.00  0.00           O  
ATOM    553  O   ROOT    1       9.836  -0.334   9.033  0.00  0.00           O  
ATOM    554  O   ROOT    1       9.145  -4.703   5.955  0.00  0.00           O  
ATOM    555  O   ROOT    1      13.275  -4.466   9.408  0.00  0.00           O  
ATOM    556  O   ROOT    1      12.584  -8.835   6.329  0.00  0.00           O  
ATOM    557  O   ROOT    1      16.715  -8.598   9.782  0.00  0.00           O  
ATOM    558  O   ROOT    1      16.023 -12.967   6.704  0.00  0.00           O  
ATOM    559  O   ROOT    1      20.154 -12.730  10.156  0.00  0.00           O  
ATOM    560  O   ROOT    1      19.462 -17.098   7.078  0.00  0.00           O  
ATOM    561  O   ROOT    1      23.593 -16.861  10.530  0.00  0.00           O  
ATOM    562  O   ROOT    1      22.902 -21.230   7.452  0.00  0.00           O  
ATOM    563  O   ROOT    1      27.032 -20.993  10.905  0.00  0.00           O  
ATOM    564  O   ROOT    1      -3.229  20.562  10.615  0.00  0.00           O  
ATOM    565  O   ROOT    1       0.902  20.798  14.067  0.00  0.00           O  
ATOM    566  O   ROOT    1       0.210  16.430  10.989  0.00  0.00           O  
ATOM    567  O   ROOT    1       4.341  16.667  14.442  0.00  0.00           O  
ATOM    568  O   ROOT    1       3.649  12.298  11.363  0.00  0.00           O  
ATOM    569  O   ROOT    1       7.780  12.535  14.816  0.00  0.00           O  
ATOM    570  O   ROOT    1       7.088   8.166  11.738  0.00  0.00           O  
ATOM    571  O   ROOT    1      11.219   8.403  15.190  0.00  0.00           O  
ATOM    572  O   ROOT    1      10.528   4.034  12.112  0.00  0.00           O  
ATOM    573  O   ROOT    1      14.659   4.271  15.564  0.00  0.00           O  
ATOM    574  O   ROOT    1      13.967  -0.097  12.486  0.00  0.00           O  
ATOM    575  O   ROOT    1      18.098   0.140  15.939  0.00  0.00           O  
ATOM    576  O   ROOT    1      17.406  -4.229  12.860  0.00  0.00           O  
ATOM    577  O   ROOT    1      21.537  -3.992  16.313  0.00  0.00           O  
ATOM    578  O   ROOT    1      20.845  -8.361  13.234  0.00  0.00           O  
ATOM    579  O   ROOT    1      24.976  -8.124  16.687  0.00  0.00           O  
ATOM    580  O   ROOT    1      24.285 -12.493  13.609  0.00  0.00           O  
ATOM    581  O   ROOT    1      28.416 -12.256  17.061  0.00  0.00           O  
ATOM    582  O   ROOT    1      27.724 -16.624  13.983  0.00  0.00           O  
ATOM    583  O   ROOT    1      31.855 -16.388  17.435  0.00  0.00           O  
ATOM    584  O   ROOT    1       1.593  25.167  17.146  0.00  0.00           O  
ATOM    585  O   ROOT    1       5.724  25.404  20.598  0.00  0.00           O  
ATOM    586  O   ROOT    1       5.032  21.035  17.520  0.00  0.00           O  
ATOM    587  O   ROOT    1       9.163  21.272  20.972  0.00  0.00           O  
ATOM    588  O   ROOT    1       8.472  16.904  17.894  0.00  0.00           O  
ATOM    589  O   ROOT    1      12.602  17.141  21.347  0.00  0.00           O  
ATOM    590  O   ROOT    1      11.911  12.772  18.268  0.00  0.00           O  
ATOM    591  O   ROOT    1      16.042  13.009  21.721  0.00  0.00           O  
ATOM    592  O   ROOT    1      15.350   8.640  18.643  0.00  0.00           O  
ATOM    593  O   ROOT    1      19.481   8.877  22.095  0.00  0.00           O  
ATOM    594  O   ROOT    1      18.789   4.508  19.017  0.00  0.00           O  
ATOM    595  O   ROOT    1      22.920   4.745  22.469  0.00  0.00           O  
ATOM    596  O   ROOT    1      22.229   0.377  19.391  0.00  0.00           O  
ATOM    597  O   ROOT    1      26.359   0.613  22.844  0.00  0.00           O  
ATOM    598  O   ROOT    1      25.668  -3.755  19.765  0.00  0.00           O  
ATOM    599  O   ROOT    1      29.799  -3.518  23.218  0.00  0.00           O  
ATOM    600  O   ROOT    1      29.107  -7.887  20.139  0.00  0.00           O  
ATOM    601  O   ROOT    1      33.238  -7.650  23.592  0.00  0.00           O  
ATOM    602  O   ROOT    1      32.546 -12.019  20.514  0.00  0.00           O  
ATOM    603  O   ROOT    1      36.677 -11.782  23.966  0.00  0.00           O  
ATOM    604  O   ROOT    1       6.416  29.773  23.676  0.00  0.00           O  
ATOM    605  O   ROOT    1      10.546  30.010  27.129  0.00  0.00           O  
ATOM    606  O   ROOT    1       9.855  25.641  24.051  0.00  0.00           O  
ATOM    607  O   ROOT    1      13.986  25.878  27.503  0.00  0.00           O  
ATOM    608  O   ROOT    1      13.294  21.509  24.425  0.00  0.00           O  
ATOM    609  O   ROOT    1      17.425  21.746  27.877  0.00  0.00           O  
ATOM    610  O   ROOT    1      16.733  17.377  24.799  0.00  0.00           O  
ATOM    611  O   ROOT    1      20.864  17.614  28.252  0.00  0.00           O  
ATOM    612  O   ROOT    1      20.173  13.246  25.173  0.00  0.00           O  
ATOM    613  O   ROOT    1      24.303  13.483  28.626  0.00  0.00           O  
ATOM    614  O   ROOT    1      23.612   9.114  25.548  0.00  0.00           O  
ATOM    615  O   ROOT    1      27.743   9.351  29.000  0.00  0.00           O  
ATOM    616  O   ROOT    1      27.051   4.982  25.922  0.00  0.00           O  
ATOM    617  O   ROOT    1      31.182   5.219  29.374  0.00  0.00           O  
ATOM    618  O   ROOT    1      30.490   0.850  26.296  0.00  0.00           O  
ATOM    619  O   ROOT    1      34.621   1.087  29.749  0.00  0.00           O  
ATOM    620  O   ROOT    1      33.930  -3.281  26.670  0.00  0.00           O  
ATOM    621  O   ROOT    1      38.060  -3.044  30.123  0.00  0.00           O  
ATOM    622  O   ROOT    1      37.369  -7.413  27.044  0.00  0.00           O  
ATOM    623  O   ROOT    1      41.500  -7.176  30.497  0.00  0.00           O  
ATOM    624  O   ROOT    1      11.238  34.378  30.207  0.00  0.00           O  
ATOM    625  O   ROOT    1      15.369  34.615  33.660  0.00  0.00           O  
ATOM    626  O   ROOT    1      14.677  30.247  30.582  0.00  0.00           O  
ATOM    627  O   ROOT    1      18.808  30.484  34.034  0.00  0.00           O  
ATOM    628  O   ROOT    1      18.116  26.115  30.956  0.00  0.00           O  
ATOM    629  O   ROOT    1      22.247  26.352  34.408  0.00  0.00           O  
ATOM    630  O   ROOT    1      21.556  21.983  31.330  0.00  0.00           O  
ATOM    631  O   ROOT    1      25.687  22.220  34.782  0.00  0.00           O  
ATOM    632  O   ROOT    1      24.995  17.851  31.704  0.00  0.00           O  
ATOM    633  O   ROOT    1      29.126  18.088  35.157  0.00  0.00           O  
ATOM    634  O   ROOT    1      28.434  13.720  32.078  0.00  0.00           O  
ATOM    635  O   ROOT    1      32.565  13.956  35.531  0.00  0.00           O  
ATOM    636  O   ROOT    1      31.873   9.588  32.453  0.00  0.00           O  
ATOM    637  O   ROOT    1      36.004   9.825  35.905  0.00  0.00           O  
ATOM    638  O   ROOT    1      35.313   5.456  32.827  0.00  0.00           O  
ATOM    639  O   ROOT    1      39.444   5.693  36.279  0.00  0.00           O  
ATOM    640  O   ROOT    1      38.752   1.324  33.201  0.00  0.00           O  
ATOM    641  O   ROOT    1      42.883   1.561  36.654  0.00  0.00           O  
ATOM    642  O   ROOT    1      42.191  -2.808  33.575  0.00  0.00           O  
ATOM    643  O   ROOT    1      46.322  -2.571  37.028  0.00  0.00           O  
ATOM    644 O1   ROOT    1      -7.148  16.228  -1.222  0.00  0.00          O1  
ATOM    645 O2   ROOT    1      -2.577  19.706  -4.826  0.00  0.00          O2  
ATOM    646 O3   ROOT    1      -2.923  17.521  -6.365  0.00  0.00          O3  
ATOM    647 O4   ROOT    1      -4.643  19.587  -6.552  0.00  0.00          O4  
ATOM    648 O5   ROOT    1      -3.029  19.570  -2.173  0.00  0.00          O5  
ATOM    649 O6   ROOT    1      -5.089  17.899  -1.697  0.00  0.00          O6  
ATOM    650 O7   ROOT    1      -5.434  15.714  -3.237  0.00  0.00          O7  
ATOM    651 O8   ROOT    1      -7.154  17.780  -3.424  0.00  0.00          O8  
ATOM    652 O9   ROOT    1      -7.946  13.907  -0.108  0.00  0.00          O9  
ATOM    653 O10  ROOT    1      -9.665  15.973  -0.295  0.00  0.00          O10  
ATOM    654 O11  ROOT    1      -7.600  16.092   1.431  0.00  0.00          O11  
ATOM    655 O13  ROOT    1      -3.017  16.464   2.231  0.00  0.00          O13  
ATOM    656 O14  ROOT    1       1.553  19.943  -1.374  0.00  0.00          O14  
ATOM    657 O15  ROOT    1       1.208  17.758  -2.913  0.00  0.00          O15  
ATOM    658 O16  ROOT    1      -0.512  19.824  -3.100  0.00  0.00          O16  
ATOM    659 O17  ROOT    1       1.102  19.807   1.279  0.00  0.00          O17  
ATOM    660 O18  ROOT    1      -0.958  18.136   1.755  0.00  0.00          O18  
ATOM    661 O19  ROOT    1      -1.304  15.951   0.216  0.00  0.00          O19  
ATOM    662 O20  ROOT    1      -3.023  18.017   0.029  0.00  0.00          O20  
ATOM    663 O21  ROOT    1      -3.815  14.144   3.345  0.00  0.00          O21  
ATOM    664 O22  ROOT    1      -5.534  16.210   3.158  0.00  0.00          O22  
ATOM    665 O23  ROOT    1      -3.469  16.329   4.884  0.00  0.00          O23  
ATOM    666 O25  ROOT    1      -3.709  12.096  -0.847  0.00  0.00          O25  
ATOM    667 O26  ROOT    1       0.862  15.574  -4.452  0.00  0.00          O26  
ATOM    668 O27  ROOT    1       0.516  13.390  -5.991  0.00  0.00          O27  
ATOM    669 O28  ROOT    1      -1.204  15.455  -6.178  0.00  0.00          O28  
ATOM    670 O29  ROOT    1       0.410  15.438  -1.799  0.00  0.00          O29  
ATOM    671 O30  ROOT    1      -1.649  13.767  -1.323  0.00  0.00          O30  
ATOM    672 O31  ROOT    1      -1.995  11.583  -2.862  0.00  0.00          O31  
ATOM    673 O32  ROOT    1      -3.715  13.648  -3.049  0.00  0.00          O32  
ATOM    674 O33  ROOT    1      -4.506   9.776   0.266  0.00  0.00          O33  
ATOM    675 O34  ROOT    1      -6.226  11.841   0.079  0.00  0.00          O34  
ATOM    676 O35  ROOT    1      -4.161  11.960   1.806  0.00  0.00          O35  
ATOM    677 O37  ROOT    1       0.422  12.333   2.605  0.00  0.00          O37  
ATOM    678 O38  ROOT    1       4.993  15.811  -0.999  0.00  0.00          O38  
ATOM    679 O39  ROOT    1       4.647  13.626  -2.538  0.00  0.00          O39  
ATOM    680 O40  ROOT    1       2.927  15.692  -2.726  0.00  0.00          O40  
ATOM    681 O41  ROOT    1       4.541  15.675   1.654  0.00  0.00          O41  
ATOM    682 O42  ROOT    1       2.481  14.004   2.129  0.00  0.00          O42  
ATOM    683 O43  ROOT    1       2.136  11.819   0.590  0.00  0.00          O43  
ATOM    684 O44  ROOT    1       0.416  13.885   0.403  0.00  0.00          O44  
ATOM    685 O45  ROOT    1      -0.376  10.012   3.719  0.00  0.00          O45  
ATOM    686 O46  ROOT    1      -2.095  12.078   3.532  0.00  0.00          O46  
ATOM    687 O47  ROOT    1      -0.030  12.197   5.258  0.00  0.00          O47  
ATOM    688 O49  ROOT    1      -0.269   7.964  -0.473  0.00  0.00          O49  
ATOM    689 O50  ROOT    1       4.301  11.442  -4.078  0.00  0.00          O50  
ATOM    690 O51  ROOT    1       3.955   9.258  -5.617  0.00  0.00          O51  
ATOM    691 O52  ROOT    1       2.236  11.324  -5.804  0.00  0.00          O52  
ATOM    692 O53  ROOT    1       3.849  11.306  -1.425  0.00  0.00          O53  
ATOM    693 O54  ROOT    1       1.790   9.635  -0.949  0.00  0.00          O54  
ATOM    694 O55  ROOT    1       1.444   7.451  -2.488  0.00  0.00          O55  
ATOM    695 O56  ROOT    1      -0.275   9.517  -2.675  0.00  0.00          O56  
ATOM    696 O57  ROOT    1      -1.067   5.644   0.641  0.00  0.00          O57  
ATOM    697 O58  ROOT    1      -2.787   7.710   0.453  0.00  0.00          O58  
ATOM    698 O59  ROOT    1      -0.721   7.828   2.180  0.00  0.00          O59  
ATOM    699 O61  ROOT    1       3.861   8.201   2.979  0.00  0.00          O61  
ATOM    700 O62  ROOT    1       8.432  11.679  -0.625  0.00  0.00          O62  
ATOM    701 O63  ROOT    1       8.086   9.495  -2.164  0.00  0.00          O63  
ATOM    702 O64  ROOT    1       6.367  11.561  -2.351  0.00  0.00          O64  
ATOM    703 O65  ROOT    1       7.980  11.543   2.028  0.00  0.00          O65  
ATOM    704 O66  ROOT    1       5.921   9.872   2.504  0.00  0.00          O66  
ATOM    705 O67  ROOT    1       5.575   7.688   0.964  0.00  0.00          O67  
ATOM    706 O68  ROOT    1       3.855   9.754   0.777  0.00  0.00          O68  
ATOM    707 O69  ROOT    1       3.064   5.881   4.093  0.00  0.00          O69  
ATOM    708 O70  ROOT    1       1.344   7.947   3.906  0.00  0.00          O70  
ATOM    709 O71  ROOT    1       3.409   8.065   5.632  0.00  0.00          O71  
ATOM    710 O73  ROOT    1       3.170   3.832  -0.099  0.00  0.00          O73  
ATOM    711 O74  ROOT    1       7.740   7.310  -3.703  0.00  0.00          O74  
ATOM    712 O75  ROOT    1       7.395   5.126  -5.243  0.00  0.00          O75  
ATOM    713 O76  ROOT    1       5.675   7.192  -5.430  0.00  0.00          O76  
ATOM    714 O77  ROOT    1       7.289   7.174  -1.051  0.00  0.00          O77  
ATOM    715 O78  ROOT    1       5.229   5.503  -0.575  0.00  0.00          O78  
ATOM    716 O79  ROOT    1       4.883   3.319  -2.114  0.00  0.00          O79  
ATOM    717 O80  ROOT    1       3.164   5.385  -2.301  0.00  0.00          O80  
ATOM    718 O81  ROOT    1       2.372   1.512   1.015  0.00  0.00          O81  
ATOM    719 O82  ROOT    1       0.652   3.578   0.828  0.00  0.00          O82  
ATOM    720 O83  ROOT    1       2.718   3.696   2.554  0.00  0.00          O83  
ATOM    721 O85  ROOT    1       7.301   4.069   3.354  0.00  0.00          O85  
ATOM    722 O86  ROOT    1      11.871   7.547  -0.251  0.00  0.00          O86  
ATOM    723 O87  ROOT    1      11.525   5.363  -1.790  0.00  0.00          O87  
ATOM    724 O88  ROOT    1       9.806   7.429  -1.977  0.00  0.00          O88  
ATOM    725 O89  ROOT    1      11.419   7.411   2.402  0.00  0.00          O89  
ATOM    726 O90  ROOT    1       9.360   5.740   2.878  0.00  0.00          O90  
ATOM    727 O91  ROOT    1       9.014   3.556   1.339  0.00  0.00          O91  
ATOM    728 O92  ROOT    1       7.295   5.622   1.152  0.00  0.00          O92  
ATOM    729 O93  ROOT    1       6.503   1.749   4.467  0.00  0.00          O93  
ATOM    730 O94  ROOT    1       4.783   3.815   4.280  0.00  0.00          O94  
ATOM    731 O95  ROOT    1       6.849   3.933   6.006  0.00  0.00          O95  
ATOM    732 O97  ROOT    1       6.609  -0.299   0.275  0.00  0.00          O97  
ATOM    733 O98  ROOT    1      11.180   3.179  -3.329  0.00  0.00          O98  
ATOM    734 O99  ROOT    1      10.834   0.994  -4.868  0.00  0.00          O99  
ATOM    735 O100 ROOT    1       9.114   3.060  -5.055  0.00  0.00          O100  
ATOM    736 O100 ROOT    1      22.217  34.115  23.398  0.00  0.00          O1000  
ATOM    737 O100 ROOT    1      23.831  34.098  27.777  0.00  0.00          O1001  
ATOM    738 O100 ROOT    1      21.771  32.426  28.253  0.00  0.00          O1002  
ATOM    739 O100 ROOT    1      21.425  30.242  26.713  0.00  0.00          O1003  
ATOM    740 O100 ROOT    1      19.706  32.308  26.526  0.00  0.00          O1004  
ATOM    741 O100 ROOT    1      18.914  28.435  29.842  0.00  0.00          O1005  
ATOM    742 O100 ROOT    1      17.194  30.501  29.655  0.00  0.00          O1006  
ATOM    743 O100 ROOT    1      19.260  30.620  31.381  0.00  0.00          O1007  
ATOM    744 O100 ROOT    1      19.020  26.387  25.650  0.00  0.00          O1009  
ATOM    745 O101 ROOT    1      10.728   3.043  -0.676  0.00  0.00          O101  
ATOM    746 O101 ROOT    1      23.591  29.865  22.046  0.00  0.00          O1010  
ATOM    747 O101 ROOT    1      23.245  27.680  20.507  0.00  0.00          O1011  
ATOM    748 O101 ROOT    1      21.525  29.746  20.319  0.00  0.00          O1012  
ATOM    749 O101 ROOT    1      23.139  29.729  24.698  0.00  0.00          O1013  
ATOM    750 O101 ROOT    1      21.080  28.058  25.174  0.00  0.00          O1014  
ATOM    751 O101 ROOT    1      20.734  25.873  23.635  0.00  0.00          O1015  
ATOM    752 O101 ROOT    1      19.014  27.939  23.448  0.00  0.00          O1016  
ATOM    753 O101 ROOT    1      18.223  24.066  26.764  0.00  0.00          O1017  
ATOM    754 O101 ROOT    1      16.503  26.132  26.577  0.00  0.00          O1018  
ATOM    755 O101 ROOT    1      18.568  26.251  28.303  0.00  0.00          O1019  
ATOM    756 O102 ROOT    1       8.668   1.372  -0.200  0.00  0.00          O102  
ATOM    757 O102 ROOT    1      23.151  26.624  29.103  0.00  0.00          O1021  
ATOM    758 O102 ROOT    1      27.722  30.102  25.498  0.00  0.00          O1022  
ATOM    759 O102 ROOT    1      27.376  27.917  23.959  0.00  0.00          O1023  
ATOM    760 O102 ROOT    1      25.656  29.983  23.772  0.00  0.00          O1024  
ATOM    761 O102 ROOT    1      27.270  29.966  28.151  0.00  0.00          O1025  
ATOM    762 O102 ROOT    1      25.210  28.295  28.627  0.00  0.00          O1026  
ATOM    763 O102 ROOT    1      24.865  26.110  27.088  0.00  0.00          O1027  
ATOM    764 O102 ROOT    1      23.145  28.176  26.901  0.00  0.00          O1028  
ATOM    765 O102 ROOT    1      22.353  24.303  30.216  0.00  0.00          O1029  
ATOM    766 O103 ROOT    1       8.323  -0.813  -1.740  0.00  0.00          O103  
ATOM    767 O103 ROOT    1      20.634  26.369  30.029  0.00  0.00          O1030  
ATOM    768 O103 ROOT    1      22.699  26.488  31.755  0.00  0.00          O1031  
ATOM    769 O103 ROOT    1      22.460  22.255  26.024  0.00  0.00          O1033  
ATOM    770 O103 ROOT    1      27.030  25.733  22.420  0.00  0.00          O1034  
ATOM    771 O103 ROOT    1      26.684  23.549  20.881  0.00  0.00          O1035  
ATOM    772 O103 ROOT    1      24.965  25.614  20.694  0.00  0.00          O1036  
ATOM    773 O103 ROOT    1      26.578  25.597  25.073  0.00  0.00          O1037  
ATOM    774 O103 ROOT    1      24.519  23.926  25.549  0.00  0.00          O1038  
ATOM    775 O103 ROOT    1      24.173  21.742  24.009  0.00  0.00          O1039  
ATOM    776 O104 ROOT    1       6.603   1.253  -1.927  0.00  0.00          O104  
ATOM    777 O104 ROOT    1      22.453  23.808  23.822  0.00  0.00          O1040  
ATOM    778 O104 ROOT    1      21.662  19.935  27.138  0.00  0.00          O1041  
ATOM    779 O104 ROOT    1      19.942  22.001  26.951  0.00  0.00          O1042  
ATOM    780 O104 ROOT    1      22.008  22.119  28.677  0.00  0.00          O1043  
ATOM    781 O104 ROOT    1      26.590  22.492  29.477  0.00  0.00          O1045  
ATOM    782 O104 ROOT    1      31.161  25.970  25.872  0.00  0.00          O1046  
ATOM    783 O104 ROOT    1      30.815  23.786  24.333  0.00  0.00          O1047  
ATOM    784 O104 ROOT    1      29.096  25.851  24.146  0.00  0.00          O1048  
ATOM    785 O104 ROOT    1      30.709  25.834  28.525  0.00  0.00          O1049  
ATOM    786 O105 ROOT    1       5.811  -2.620   1.389  0.00  0.00          O105  
ATOM    787 O105 ROOT    1      28.650  24.163  29.001  0.00  0.00          O1050  
ATOM    788 O105 ROOT    1      28.304  21.979  27.462  0.00  0.00          O1051  
ATOM    789 O105 ROOT    1      26.584  24.044  27.275  0.00  0.00          O1052  
ATOM    790 O105 ROOT    1      25.793  20.172  30.591  0.00  0.00          O1053  
ATOM    791 O105 ROOT    1      24.073  22.237  30.403  0.00  0.00          O1054  
ATOM    792 O105 ROOT    1      26.138  22.356  32.130  0.00  0.00          O1055  
ATOM    793 O105 ROOT    1      25.899  18.123  26.399  0.00  0.00          O1057  
ATOM    794 O105 ROOT    1      30.469  21.601  22.794  0.00  0.00          O1058  
ATOM    795 O105 ROOT    1      30.124  19.417  21.255  0.00  0.00          O1059  
ATOM    796 O106 ROOT    1       4.092  -0.554   1.202  0.00  0.00          O106  
ATOM    797 O106 ROOT    1      28.404  21.483  21.068  0.00  0.00          O1060  
ATOM    798 O106 ROOT    1      30.017  21.465  25.447  0.00  0.00          O1061  
ATOM    799 O106 ROOT    1      27.958  19.794  25.923  0.00  0.00          O1062  
ATOM    800 O106 ROOT    1      27.612  17.610  24.384  0.00  0.00          O1063  
ATOM    801 O106 ROOT    1      25.893  19.676  24.196  0.00  0.00          O1064  
ATOM    802 O106 ROOT    1      25.101  15.803  27.512  0.00  0.00          O1065  
ATOM    803 O106 ROOT    1      23.381  17.869  27.325  0.00  0.00          O1066  
ATOM    804 O106 ROOT    1      25.447  17.987  29.051  0.00  0.00          O1067  
ATOM    805 O106 ROOT    1      30.030  18.360  29.851  0.00  0.00          O1069  
ATOM    806 O107 ROOT    1       6.157  -0.435   2.928  0.00  0.00          O107  
ATOM    807 O107 ROOT    1      34.600  21.838  26.247  0.00  0.00          O1070  
ATOM    808 O107 ROOT    1      34.254  19.654  24.707  0.00  0.00          O1071  
ATOM    809 O107 ROOT    1      32.535  21.720  24.520  0.00  0.00          O1072  
ATOM    810 O107 ROOT    1      34.148  21.702  28.899  0.00  0.00          O1073  
ATOM    811 O107 ROOT    1      32.089  20.031  29.375  0.00  0.00          O1074  
ATOM    812 O107 ROOT    1      31.743  17.847  27.836  0.00  0.00          O1075  
ATOM    813 O107 ROOT    1      30.024  19.913  27.649  0.00  0.00          O1076  
ATOM    814 O107 ROOT    1      29.232  16.040  30.965  0.00  0.00          O1077  
ATOM    815 O107 ROOT    1      27.512  18.106  30.778  0.00  0.00          O1078  
ATOM    816 O107 ROOT    1      29.578  18.224  32.504  0.00  0.00          O1079  
ATOM    817 O108 ROOT    1      29.338  13.991  26.773  0.00  0.00          O1081  
ATOM    818 O108 ROOT    1      33.909  17.469  23.168  0.00  0.00          O1082  
ATOM    819 O108 ROOT    1      33.563  15.285  21.629  0.00  0.00          O1083  
ATOM    820 O108 ROOT    1      31.843  17.351  21.442  0.00  0.00          O1084  
ATOM    821 O108 ROOT    1      33.457  17.334  25.821  0.00  0.00          O1085  
ATOM    822 O108 ROOT    1      31.397  15.662  26.297  0.00  0.00          O1086  
ATOM    823 O108 ROOT    1      31.052  13.478  24.758  0.00  0.00          O1087  
ATOM    824 O108 ROOT    1      29.332  15.544  24.571  0.00  0.00          O1088  
ATOM    825 O108 ROOT    1      28.540  11.671  27.886  0.00  0.00          O1089  
ATOM    826 O109 ROOT    1      10.740  -0.063   3.728  0.00  0.00          O109  
ATOM    827 O109 ROOT    1      26.821  13.737  27.699  0.00  0.00          O1090  
ATOM    828 O109 ROOT    1      28.886  13.855  29.426  0.00  0.00          O1091  
ATOM    829 O109 ROOT    1      33.469  14.228  30.225  0.00  0.00          O1093  
ATOM    830 O109 ROOT    1      38.039  17.706  26.621  0.00  0.00          O1094  
ATOM    831 O109 ROOT    1      37.694  15.522  25.082  0.00  0.00          O1095  
ATOM    832 O109 ROOT    1      35.974  17.588  24.895  0.00  0.00          O1096  
ATOM    833 O109 ROOT    1      37.588  17.570  29.274  0.00  0.00          O1097  
ATOM    834 O109 ROOT    1      35.528  15.899  29.749  0.00  0.00          O1098  
ATOM    835 O109 ROOT    1      35.182  13.715  28.210  0.00  0.00          O1099  
ATOM    836 O110 ROOT    1      15.310   3.415   0.123  0.00  0.00          O110  
ATOM    837 O110 ROOT    1      33.463  15.781  28.023  0.00  0.00          O1100  
ATOM    838 O110 ROOT    1      32.671  11.908  31.339  0.00  0.00          O1101  
ATOM    839 O110 ROOT    1      30.951  13.974  31.152  0.00  0.00          O1102  
ATOM    840 O110 ROOT    1      33.017  14.092  32.878  0.00  0.00          O1103  
ATOM    841 O110 ROOT    1      32.777   9.860  27.147  0.00  0.00          O1105  
ATOM    842 O110 ROOT    1      37.348  13.338  23.543  0.00  0.00          O1106  
ATOM    843 O110 ROOT    1      37.002  11.153  22.003  0.00  0.00          O1107  
ATOM    844 O110 ROOT    1      35.282  13.219  21.816  0.00  0.00          O1108  
ATOM    845 O110 ROOT    1      36.896  13.202  26.195  0.00  0.00          O1109  
ATOM    846 O111 ROOT    1      14.965   1.231  -1.416  0.00  0.00          O111  
ATOM    847 O111 ROOT    1      34.837  11.531  26.671  0.00  0.00          O1110  
ATOM    848 O111 ROOT    1      34.491   9.346  25.132  0.00  0.00          O1111  
ATOM    849 O111 ROOT    1      32.771  11.412  24.945  0.00  0.00          O1112  
ATOM    850 O111 ROOT    1      31.980   7.539  28.261  0.00  0.00          O1113  
ATOM    851 O111 ROOT    1      30.260   9.605  28.074  0.00  0.00          O1114  
ATOM    852 O111 ROOT    1      32.325   9.724  29.800  0.00  0.00          O1115  
ATOM    853 O111 ROOT    1      36.908  10.097  30.600  0.00  0.00          O1117  
ATOM    854 O111 ROOT    1      41.479  13.575  26.995  0.00  0.00          O1118  
ATOM    855 O111 ROOT    1      41.133  11.390  25.456  0.00  0.00          O1119  
ATOM    856 O112 ROOT    1      13.245   3.297  -1.603  0.00  0.00          O112  
ATOM    857 O112 ROOT    1      39.413  13.456  25.269  0.00  0.00          O1120  
ATOM    858 O112 ROOT    1      41.027  13.439  29.648  0.00  0.00          O1121  
ATOM    859 O112 ROOT    1      38.967  11.768  30.124  0.00  0.00          O1122  
ATOM    860 O112 ROOT    1      38.622   9.583  28.585  0.00  0.00          O1123  
ATOM    861 O112 ROOT    1      36.902  11.649  28.397  0.00  0.00          O1124  
ATOM    862 O112 ROOT    1      36.110   7.776  31.713  0.00  0.00          O1125  
ATOM    863 O112 ROOT    1      34.391   9.842  31.526  0.00  0.00          O1126  
ATOM    864 O112 ROOT    1      36.456   9.961  33.252  0.00  0.00          O1127  
ATOM    865 O112 ROOT    1      36.217   5.728  27.521  0.00  0.00          O1129  
ATOM    866 O113 ROOT    1      14.859   3.280   2.776  0.00  0.00          O113  
ATOM    867 O113 ROOT    1      40.787   9.206  23.917  0.00  0.00          O1130  
ATOM    868 O113 ROOT    1      40.441   7.021  22.378  0.00  0.00          O1131  
ATOM    869 O113 ROOT    1      38.722   9.087  22.190  0.00  0.00          O1132  
ATOM    870 O113 ROOT    1      40.335   9.070  26.570  0.00  0.00          O1133  
ATOM    871 O113 ROOT    1      38.276   7.399  27.045  0.00  0.00          O1134  
ATOM    872 O113 ROOT    1      37.930   5.215  25.506  0.00  0.00          O1135  
ATOM    873 O113 ROOT    1      36.211   7.280  25.319  0.00  0.00          O1136  
ATOM    874 O113 ROOT    1      35.419   3.408  28.635  0.00  0.00          O1137  
ATOM    875 O113 ROOT    1      33.699   5.473  28.448  0.00  0.00          O1138  
ATOM    876 O113 ROOT    1      35.765   5.592  30.174  0.00  0.00          O1139  
ATOM    877 O114 ROOT    1      12.799   1.608   3.252  0.00  0.00          O114  
ATOM    878 O114 ROOT    1      40.347   5.965  30.974  0.00  0.00          O1141  
ATOM    879 O114 ROOT    1      44.918   9.443  27.369  0.00  0.00          O1142  
ATOM    880 O114 ROOT    1      44.572   7.258  25.830  0.00  0.00          O1143  
ATOM    881 O114 ROOT    1      42.853   9.324  25.643  0.00  0.00          O1144  
ATOM    882 O114 ROOT    1      44.466   9.307  30.022  0.00  0.00          O1145  
ATOM    883 O114 ROOT    1      42.407   7.636  30.498  0.00  0.00          O1146  
ATOM    884 O114 ROOT    1      42.061   5.451  28.959  0.00  0.00          O1147  
ATOM    885 O114 ROOT    1      40.341   7.517  28.772  0.00  0.00          O1148  
ATOM    886 O114 ROOT    1      39.550   3.644  32.087  0.00  0.00          O1149  
ATOM    887 O115 ROOT    1      12.453  -0.576   1.713  0.00  0.00          O115  
ATOM    888 O115 ROOT    1      37.830   5.710  31.900  0.00  0.00          O1150  
ATOM    889 O115 ROOT    1      39.896   5.829  33.627  0.00  0.00          O1151  
ATOM    890 O115 ROOT    1      39.656   1.596  27.895  0.00  0.00          O1153  
ATOM    891 O115 ROOT    1      44.226   5.074  24.291  0.00  0.00          O1154  
ATOM    892 O115 ROOT    1      43.881   2.890  22.752  0.00  0.00          O1155  
ATOM    893 O115 ROOT    1      42.161   4.956  22.565  0.00  0.00          O1156  
ATOM    894 O115 ROOT    1      43.775   4.938  26.944  0.00  0.00          O1157  
ATOM    895 O115 ROOT    1      41.715   3.267  27.420  0.00  0.00          O1158  
ATOM    896 O115 ROOT    1      41.369   1.083  25.880  0.00  0.00          O1159  
ATOM    897 O116 ROOT    1      10.734   1.490   1.526  0.00  0.00          O116  
ATOM    898 O116 ROOT    1      39.650   3.149  25.693  0.00  0.00          O1160  
ATOM    899 O116 ROOT    1      38.858  -0.724  29.009  0.00  0.00          O1161  
ATOM    900 O116 ROOT    1      37.138   1.342  28.822  0.00  0.00          O1162  
ATOM    901 O116 ROOT    1      39.204   1.460  30.548  0.00  0.00          O1163  
ATOM    902 O116 ROOT    1      43.787   1.833  31.348  0.00  0.00          O1165  
ATOM    903 O116 ROOT    1      48.357   5.311  27.744  0.00  0.00          O1166  
ATOM    904 O116 ROOT    1      48.011   3.127  26.204  0.00  0.00          O1167  
ATOM    905 O116 ROOT    1      46.292   5.192  26.017  0.00  0.00          O1168  
ATOM    906 O116 ROOT    1      47.905   5.175  30.396  0.00  0.00          O1169  
ATOM    907 O117 ROOT    1       9.942  -2.383   4.842  0.00  0.00          O117  
ATOM    908 O117 ROOT    1      45.846   3.504  30.872  0.00  0.00          O1170  
ATOM    909 O117 ROOT    1      45.500   1.320  29.333  0.00  0.00          O1171  
ATOM    910 O117 ROOT    1      43.781   3.386  29.146  0.00  0.00          O1172  
ATOM    911 O117 ROOT    1      42.989  -0.487  32.462  0.00  0.00          O1173  
ATOM    912 O117 ROOT    1      41.269   1.579  32.275  0.00  0.00          O1174  
ATOM    913 O117 ROOT    1      43.335   1.697  34.001  0.00  0.00          O1175  
ATOM    914 O117 ROOT    1      43.095  -2.536  28.270  0.00  0.00          O1177  
ATOM    915 O117 ROOT    1      47.666   0.942  24.665  0.00  0.00          O1178  
ATOM    916 O117 ROOT    1      47.320  -1.242  23.126  0.00  0.00          O1179  
ATOM    917 O118 ROOT    1       8.223  -0.317   4.654  0.00  0.00          O118  
ATOM    918 O118 ROOT    1      45.600   0.824  22.939  0.00  0.00          O1180  
ATOM    919 O118 ROOT    1      47.214   0.806  27.318  0.00  0.00          O1181  
ATOM    920 O118 ROOT    1      45.154  -0.865  27.794  0.00  0.00          O1182  
ATOM    921 O118 ROOT    1      44.809  -3.049  26.255  0.00  0.00          O1183  
ATOM    922 O118 ROOT    1      43.089  -0.983  26.068  0.00  0.00          O1184  
ATOM    923 O118 ROOT    1      42.297  -4.856  29.383  0.00  0.00          O1185  
ATOM    924 O118 ROOT    1      40.578  -2.790  29.196  0.00  0.00          O1186  
ATOM    925 O118 ROOT    1      42.643  -2.672  30.923  0.00  0.00          O1187  
ATOM    926 O118 ROOT    1      47.226  -2.299  31.722  0.00  0.00          O1189  
ATOM    927 O119 ROOT    1      10.288  -0.198   6.381  0.00  0.00          O119  
ATOM    928 O119 ROOT    1      51.797   1.179  28.118  0.00  0.00          O1190  
ATOM    929 O119 ROOT    1      51.451  -1.005  26.579  0.00  0.00          O1191  
ATOM    930 O119 ROOT    1      49.731   1.061  26.391  0.00  0.00          O1192  
ATOM    931 O119 ROOT    1      51.345   1.043  30.771  0.00  0.00          O1193  
ATOM    932 O119 ROOT    1      49.285  -0.628  31.246  0.00  0.00          O1194  
ATOM    933 O119 ROOT    1      48.939  -2.812  29.707  0.00  0.00          O1195  
ATOM    934 O119 ROOT    1      47.220  -0.746  29.520  0.00  0.00          O1196  
ATOM    935 O119 ROOT    1      46.428  -4.619  32.836  0.00  0.00          O1197  
ATOM    936 O119 ROOT    1      44.709  -2.553  32.649  0.00  0.00          O1198  
ATOM    937 O119 ROOT    1      46.774  -2.435  34.375  0.00  0.00          O1199  
ATOM    938 O120 ROOT    1      -2.125  19.842  -7.479  0.00  0.00          O1201  
ATOM    939 O120 ROOT    1       2.005  20.078  -4.026  0.00  0.00          O1202  
ATOM    940 O120 ROOT    1       1.314  15.710  -7.105  0.00  0.00          O1203  
ATOM    941 O120 ROOT    1       5.445  15.947  -3.652  0.00  0.00          O1204  
ATOM    942 O120 ROOT    1       4.753  11.578  -6.730  0.00  0.00          O1205  
ATOM    943 O120 ROOT    1       8.884  11.815  -3.278  0.00  0.00          O1206  
ATOM    944 O120 ROOT    1       8.192   7.446  -6.356  0.00  0.00          O1207  
ATOM    945 O120 ROOT    1      12.323   7.683  -2.904  0.00  0.00          O1208  
ATOM    946 O120 ROOT    1      11.631   3.314  -5.982  0.00  0.00          O1209  
ATOM    947 O121 ROOT    1      10.048  -4.431   0.650  0.00  0.00          O121  
ATOM    948 O121 ROOT    1      15.762   3.551  -2.529  0.00  0.00          O1210  
ATOM    949 O121 ROOT    1      15.071  -0.817  -5.608  0.00  0.00          O1211  
ATOM    950 O121 ROOT    1      19.202  -0.580  -2.155  0.00  0.00          O1212  
ATOM    951 O121 ROOT    1      18.510  -4.949  -5.234  0.00  0.00          O1213  
ATOM    952 O121 ROOT    1      22.641  -4.712  -1.781  0.00  0.00          O1214  
ATOM    953 O121 ROOT    1      21.949  -9.081  -4.859  0.00  0.00          O1215  
ATOM    954 O121 ROOT    1      26.080  -8.844  -1.407  0.00  0.00          O1216  
ATOM    955 O121 ROOT    1      25.389 -13.213  -4.485  0.00  0.00          O1217  
ATOM    956 O121 ROOT    1      29.519 -12.976  -1.033  0.00  0.00          O1218  
ATOM    957 O121 ROOT    1      28.828 -17.344  -4.111  0.00  0.00          O1219  
ATOM    958 O122 ROOT    1      14.619  -0.953  -2.955  0.00  0.00          O122  
ATOM    959 O122 ROOT    1      32.959 -17.107  -0.658  0.00  0.00          O1220  
ATOM    960 O122 ROOT    1       2.697  24.447  -0.948  0.00  0.00          O1221  
ATOM    961 O122 ROOT    1       6.828  24.684   2.505  0.00  0.00          O1222  
ATOM    962 O122 ROOT    1       6.136  20.315  -0.574  0.00  0.00          O1223  
ATOM    963 O122 ROOT    1      10.267  20.552   2.879  0.00  0.00          O1224  
ATOM    964 O122 ROOT    1       9.575  16.184  -0.200  0.00  0.00          O1225  
ATOM    965 O122 ROOT    1      13.706  16.421   3.253  0.00  0.00          O1226  
ATOM    966 O122 ROOT    1      13.015  12.052   0.175  0.00  0.00          O1227  
ATOM    967 O122 ROOT    1      17.146  12.289   3.627  0.00  0.00          O1228  
ATOM    968 O122 ROOT    1      16.454   7.920   0.549  0.00  0.00          O1229  
ATOM    969 O123 ROOT    1      14.273  -3.138  -4.494  0.00  0.00          O123  
ATOM    970 O123 ROOT    1      20.585   8.157   4.001  0.00  0.00          O1230  
ATOM    971 O123 ROOT    1      19.893   3.788   0.923  0.00  0.00          O1231  
ATOM    972 O123 ROOT    1      24.024   4.025   4.376  0.00  0.00          O1232  
ATOM    973 O123 ROOT    1      23.332  -0.343   1.297  0.00  0.00          O1233  
ATOM    974 O123 ROOT    1      27.463  -0.106   4.750  0.00  0.00          O1234  
ATOM    975 O123 ROOT    1      26.772  -4.475   1.672  0.00  0.00          O1235  
ATOM    976 O123 ROOT    1      30.903  -4.238   5.124  0.00  0.00          O1236  
ATOM    977 O123 ROOT    1      30.211  -8.607   2.046  0.00  0.00          O1237  
ATOM    978 O123 ROOT    1      34.342  -8.370   5.498  0.00  0.00          O1238  
ATOM    979 O123 ROOT    1      33.650 -12.739   2.420  0.00  0.00          O1239  
ATOM    980 O124 ROOT    1      12.553  -1.072  -4.681  0.00  0.00          O124  
ATOM    981 O124 ROOT    1      37.781 -12.502   5.872  0.00  0.00          O1240  
ATOM    982 O124 ROOT    1       7.519  29.053   5.583  0.00  0.00          O1241  
ATOM    983 O124 ROOT    1      11.650  29.290   9.035  0.00  0.00          O1242  
ATOM    984 O124 ROOT    1      10.959  24.921   5.957  0.00  0.00          O1243  
ATOM    985 O124 ROOT    1      15.089  25.158   9.410  0.00  0.00          O1244  
ATOM    986 O124 ROOT    1      14.398  20.789   6.331  0.00  0.00          O1245  
ATOM    987 O124 ROOT    1      18.529  21.026   9.784  0.00  0.00          O1246  
ATOM    988 O124 ROOT    1      17.837  16.658   6.705  0.00  0.00          O1247  
ATOM    989 O124 ROOT    1      21.968  16.894  10.158  0.00  0.00          O1248  
ATOM    990 O124 ROOT    1      21.276  12.526   7.080  0.00  0.00          O1249  
ATOM    991 O125 ROOT    1      14.167  -1.089  -0.302  0.00  0.00          O125  
ATOM    992 O125 ROOT    1      25.407  12.763  10.532  0.00  0.00          O1250  
ATOM    993 O125 ROOT    1      24.716   8.394   7.454  0.00  0.00          O1251  
ATOM    994 O125 ROOT    1      28.847   8.631  10.906  0.00  0.00          O1252  
ATOM    995 O125 ROOT    1      28.155   4.262   7.828  0.00  0.00          O1253  
ATOM    996 O125 ROOT    1      32.286   4.499  11.281  0.00  0.00          O1254  
ATOM    997 O125 ROOT    1      31.594   0.130   8.202  0.00  0.00          O1255  
ATOM    998 O125 ROOT    1      35.725   0.367  11.655  0.00  0.00          O1256  
ATOM    999 O125 ROOT    1      35.033  -4.001   8.577  0.00  0.00          O1257  
ATOM   1000 O125 ROOT    1      39.164  -3.764  12.029  0.00  0.00          O1258  
ATOM   1001 O125 ROOT    1      38.473  -8.133   8.951  0.00  0.00          O1259  
ATOM   1002 O126 ROOT    1      12.108  -2.760   0.174  0.00  0.00          O126  
ATOM   1003 O126 ROOT    1      42.604  -7.896  12.403  0.00  0.00          O1260  
ATOM   1004 O126 ROOT    1      12.342  33.659  12.114  0.00  0.00          O1261  
ATOM   1005 O126 ROOT    1      16.473  33.895  15.566  0.00  0.00          O1262  
ATOM   1006 O126 ROOT    1      15.781  29.527  12.488  0.00  0.00          O1263  
ATOM   1007 O126 ROOT    1      19.912  29.764  15.940  0.00  0.00          O1264  
ATOM   1008 O126 ROOT    1      19.220  25.395  12.862  0.00  0.00          O1265  
ATOM   1009 O126 ROOT    1      23.351  25.632  16.315  0.00  0.00          O1266  
ATOM   1010 O126 ROOT    1      22.660  21.263  13.236  0.00  0.00          O1267  
ATOM   1011 O126 ROOT    1      26.790  21.500  16.689  0.00  0.00          O1268  
ATOM   1012 O126 ROOT    1      26.099  17.131  13.611  0.00  0.00          O1269  
ATOM   1013 O127 ROOT    1      11.762  -4.944  -1.365  0.00  0.00          O127  
ATOM   1014 O127 ROOT    1      30.230  17.368  17.063  0.00  0.00          O1270  
ATOM   1015 O127 ROOT    1      29.538  13.000  13.985  0.00  0.00          O1271  
ATOM   1016 O127 ROOT    1      33.669  13.237  17.437  0.00  0.00          O1272  
ATOM   1017 O127 ROOT    1      32.977   8.868  14.359  0.00  0.00          O1273  
ATOM   1018 O127 ROOT    1      37.108   9.105  17.811  0.00  0.00          O1274  
ATOM   1019 O127 ROOT    1      36.417   4.736  14.733  0.00  0.00          O1275  
ATOM   1020 O127 ROOT    1      40.547   4.973  18.186  0.00  0.00          O1276  
ATOM   1021 O127 ROOT    1      39.856   0.604  15.107  0.00  0.00          O1277  
ATOM   1022 O127 ROOT    1      43.987   0.841  18.560  0.00  0.00          O1278  
ATOM   1023 O127 ROOT    1      43.295  -3.527  15.482  0.00  0.00          O1279  
ATOM   1024 O128 ROOT    1      10.042  -2.879  -1.553  0.00  0.00          O128  
ATOM   1025 O128 ROOT    1      47.426  -3.291  18.934  0.00  0.00          O1280  
ATOM   1026 O128 ROOT    1      17.164  38.264  18.644  0.00  0.00          O1281  
ATOM   1027 O128 ROOT    1      21.295  38.501  22.097  0.00  0.00          O1282  
ATOM   1028 O128 ROOT    1      20.603  34.132  19.019  0.00  0.00          O1283  
ATOM   1029 O128 ROOT    1      24.734  34.369  22.471  0.00  0.00          O1284  
ATOM   1030 O128 ROOT    1      24.043  30.001  19.393  0.00  0.00          O1285  
ATOM   1031 O128 ROOT    1      28.174  30.238  22.845  0.00  0.00          O1286  
ATOM   1032 O128 ROOT    1      27.482  25.869  19.767  0.00  0.00          O1287  
ATOM   1033 O128 ROOT    1      31.613  26.106  23.220  0.00  0.00          O1288  
ATOM   1034 O128 ROOT    1      30.921  21.737  20.141  0.00  0.00          O1289  
ATOM   1035 O129 ROOT    1       9.251  -6.751   1.763  0.00  0.00          O129  
ATOM   1036 O129 ROOT    1      35.052  21.974  23.594  0.00  0.00          O1290  
ATOM   1037 O129 ROOT    1      34.360  17.605  20.515  0.00  0.00          O1291  
ATOM   1038 O129 ROOT    1      38.491  17.842  23.968  0.00  0.00          O1292  
ATOM   1039 O129 ROOT    1      37.800  13.473  20.890  0.00  0.00          O1293  
ATOM   1040 O129 ROOT    1      41.931  13.710  24.342  0.00  0.00          O1294  
ATOM   1041 O129 ROOT    1      41.239   9.342  21.264  0.00  0.00          O1295  
ATOM   1042 O129 ROOT    1      45.370   9.579  24.717  0.00  0.00          O1296  
ATOM   1043 O129 ROOT    1      44.678   5.210  21.638  0.00  0.00          O1297  
ATOM   1044 O129 ROOT    1      48.809   5.447  25.091  0.00  0.00          O1298  
ATOM   1045 O129 ROOT    1      48.117   1.078  22.012  0.00  0.00          O1299  
ATOM   1046 O130 ROOT    1       7.531  -4.686   1.576  0.00  0.00          O130  
ATOM   1047 O130 ROOT    1      52.248   1.315  25.465  0.00  0.00          O1300  
ATOM   1048 O131 ROOT    1       9.596  -4.567   3.302  0.00  0.00          O131  
ATOM   1049 O133 ROOT    1      14.179  -4.194   4.102  0.00  0.00          O133  
ATOM   1050 O134 ROOT    1      18.750  -0.716   0.498  0.00  0.00          O134  
ATOM   1051 O135 ROOT    1      18.404  -2.901  -1.042  0.00  0.00          O135  
ATOM   1052 O136 ROOT    1      16.684  -0.835  -1.229  0.00  0.00          O136  
ATOM   1053 O137 ROOT    1      18.298  -0.852   3.150  0.00  0.00          O137  
ATOM   1054 O138 ROOT    1      16.239  -2.523   3.626  0.00  0.00          O138  
ATOM   1055 O139 ROOT    1      15.893  -4.708   2.087  0.00  0.00          O139  
ATOM   1056 O140 ROOT    1      14.173  -2.642   1.900  0.00  0.00          O140  
ATOM   1057 O141 ROOT    1      13.381  -6.515   5.216  0.00  0.00          O141  
ATOM   1058 O142 ROOT    1      11.662  -4.449   5.029  0.00  0.00          O142  
ATOM   1059 O143 ROOT    1      13.727  -4.330   6.755  0.00  0.00          O143  
ATOM   1060 O145 ROOT    1      13.488  -8.563   1.024  0.00  0.00          O145  
ATOM   1061 O146 ROOT    1      18.058  -5.085  -2.581  0.00  0.00          O146  
ATOM   1062 O147 ROOT    1      17.712  -7.269  -4.120  0.00  0.00          O147  
ATOM   1063 O148 ROOT    1      15.993  -5.203  -4.307  0.00  0.00          O148  
ATOM   1064 O149 ROOT    1      17.606  -5.221   0.072  0.00  0.00          O149  
ATOM   1065 O150 ROOT    1      15.547  -6.892   0.548  0.00  0.00          O150  
ATOM   1066 O151 ROOT    1      15.201  -9.076  -0.991  0.00  0.00          O151  
ATOM   1067 O152 ROOT    1      13.482  -7.010  -1.178  0.00  0.00          O152  
ATOM   1068 O153 ROOT    1      12.690 -10.883   2.137  0.00  0.00          O153  
ATOM   1069 O154 ROOT    1      10.970  -8.817   1.950  0.00  0.00          O154  
ATOM   1070 O155 ROOT    1      13.036  -8.699   3.677  0.00  0.00          O155  
ATOM   1071 O157 ROOT    1      17.618  -8.326   4.476  0.00  0.00          O157  
ATOM   1072 O158 ROOT    1      22.189  -4.848   0.872  0.00  0.00          O158  
ATOM   1073 O159 ROOT    1      21.843  -7.032  -0.667  0.00  0.00          O159  
ATOM   1074 O160 ROOT    1      20.124  -4.966  -0.854  0.00  0.00          O160  
ATOM   1075 O161 ROOT    1      21.737  -4.984   3.525  0.00  0.00          O161  
ATOM   1076 O162 ROOT    1      19.678  -6.655   4.000  0.00  0.00          O162  
ATOM   1077 O163 ROOT    1      19.332  -8.839   2.461  0.00  0.00          O163  
ATOM   1078 O164 ROOT    1      17.612  -6.773   2.274  0.00  0.00          O164  
ATOM   1079 O165 ROOT    1      16.821 -10.646   5.590  0.00  0.00          O165  
ATOM   1080 O166 ROOT    1      15.101  -8.580   5.403  0.00  0.00          O166  
ATOM   1081 O167 ROOT    1      17.166  -8.462   7.129  0.00  0.00          O167  
ATOM   1082 O169 ROOT    1      16.927 -12.695   1.398  0.00  0.00          O169  
ATOM   1083 O170 ROOT    1      21.497  -9.217  -2.206  0.00  0.00          O170  
ATOM   1084 O171 ROOT    1      21.152 -11.401  -3.746  0.00  0.00          O171  
ATOM   1085 O172 ROOT    1      19.432  -9.335  -3.933  0.00  0.00          O172  
ATOM   1086 O173 ROOT    1      21.046  -9.353   0.446  0.00  0.00          O173  
ATOM   1087 O174 ROOT    1      18.986 -11.024   0.922  0.00  0.00          O174  
ATOM   1088 O175 ROOT    1      18.640 -13.208  -0.617  0.00  0.00          O175  
ATOM   1089 O176 ROOT    1      16.921 -11.142  -0.804  0.00  0.00          O176  
ATOM   1090 O177 ROOT    1      16.129 -15.015   2.512  0.00  0.00          O177  
ATOM   1091 O178 ROOT    1      14.410 -12.949   2.325  0.00  0.00          O178  
ATOM   1092 O179 ROOT    1      16.475 -12.831   4.051  0.00  0.00          O179  
ATOM   1093 O181 ROOT    1      21.058 -12.458   4.850  0.00  0.00          O181  
ATOM   1094 O182 ROOT    1      25.628  -8.980   1.246  0.00  0.00          O182  
ATOM   1095 O183 ROOT    1      25.282 -11.164  -0.293  0.00  0.00          O183  
ATOM   1096 O184 ROOT    1      23.563  -9.098  -0.480  0.00  0.00          O184  
ATOM   1097 O185 ROOT    1      25.176  -9.116   3.899  0.00  0.00          O185  
ATOM   1098 O186 ROOT    1      23.117 -10.787   4.375  0.00  0.00          O186  
ATOM   1099 O187 ROOT    1      22.771 -12.971   2.836  0.00  0.00          O187  
ATOM   1100 O188 ROOT    1      21.052 -10.905   2.648  0.00  0.00          O188  
ATOM   1101 O189 ROOT    1      20.260 -14.778   5.964  0.00  0.00          O189  
ATOM   1102 O190 ROOT    1      18.540 -12.712   5.777  0.00  0.00          O190  
ATOM   1103 O191 ROOT    1      20.606 -12.594   7.503  0.00  0.00          O191  
ATOM   1104 O193 ROOT    1      20.366 -16.827   1.772  0.00  0.00          O193  
ATOM   1105 O194 ROOT    1      24.937 -13.349  -1.832  0.00  0.00          O194  
ATOM   1106 O195 ROOT    1      24.591 -15.533  -3.371  0.00  0.00          O195  
ATOM   1107 O196 ROOT    1      22.871 -13.467  -3.559  0.00  0.00          O196  
ATOM   1108 O197 ROOT    1      24.485 -13.484   0.821  0.00  0.00          O197  
ATOM   1109 O198 ROOT    1      22.425 -15.156   1.296  0.00  0.00          O198  
ATOM   1110 O199 ROOT    1      22.080 -17.340  -0.243  0.00  0.00          O199  
ATOM   1111 O200 ROOT    1      20.360 -15.274  -0.430  0.00  0.00          O200  
ATOM   1112 Si   ROOT    1      -3.067  19.164  -6.306  0.00  0.00          Si  
ATOM   1113 Si   ROOT    1      -2.088  20.247  -3.346  0.00  0.00          Si  
ATOM   1114 Si   ROOT    1      -3.971  18.892  -1.000  0.00  0.00          Si  
ATOM   1115 Si   ROOT    1      -6.206  16.905  -2.395  0.00  0.00          Si  
ATOM   1116 Si   ROOT    1      -8.090  15.550  -0.048  0.00  0.00          Si  
ATOM   1117 Si   ROOT    1      -7.110  16.633   2.911  0.00  0.00          Si  
ATOM   1118 Si   ROOT    1       1.064  19.401  -2.853  0.00  0.00          Si  
ATOM   1119 Si   ROOT    1       2.043  20.484   0.106  0.00  0.00          Si  
ATOM   1120 Si   ROOT    1       0.160  19.129   2.453  0.00  0.00          Si  
ATOM   1121 Si   ROOT    1      -2.075  17.142   1.058  0.00  0.00          Si  
ATOM   1122 Si   ROOT    1      -3.959  15.787   3.404  0.00  0.00          Si  
ATOM   1123 Si   ROOT    1      -2.979  16.870   6.363  0.00  0.00          Si  
ATOM   1124 Si   ROOT    1       0.372  15.032  -5.931  0.00  0.00          Si  
ATOM   1125 Si   ROOT    1       1.352  16.116  -2.972  0.00  0.00          Si  
ATOM   1126 Si   ROOT    1      -0.532  14.760  -0.626  0.00  0.00          Si  
ATOM   1127 Si   ROOT    1      -2.767  12.773  -2.021  0.00  0.00          Si  
ATOM   1128 Si   ROOT    1      -4.650  11.418   0.326  0.00  0.00          Si  
ATOM   1129 Si   ROOT    1      -3.671  12.502   3.285  0.00  0.00          Si  
ATOM   1130 Si   ROOT    1       4.503  15.269  -2.479  0.00  0.00          Si  
ATOM   1131 Si   ROOT    1       5.483  16.353   0.480  0.00  0.00          Si  
ATOM   1132 Si   ROOT    1       3.599  14.997   2.827  0.00  0.00          Si  
ATOM   1133 Si   ROOT    1       1.364  13.010   1.432  0.00  0.00          Si  
ATOM   1134 Si   ROOT    1      -0.520  11.655   3.778  0.00  0.00          Si  
ATOM   1135 Si   ROOT    1       0.460  12.739   6.738  0.00  0.00          Si  
ATOM   1136 Si   ROOT    1       3.811  10.900  -5.557  0.00  0.00          Si  
ATOM   1137 Si   ROOT    1       4.791  11.984  -2.598  0.00  0.00          Si  
ATOM   1138 Si   ROOT    1       2.908  10.629  -0.252  0.00  0.00          Si  
ATOM   1139 Si   ROOT    1       0.672   8.642  -1.646  0.00  0.00          Si  
ATOM   1140 Si   ROOT    1      -1.211   7.286   0.700  0.00  0.00          Si  
ATOM   1141 Si   ROOT    1      -0.231   8.370   3.659  0.00  0.00          Si  
ATOM   1142 Si   ROOT    1       7.942  11.137  -2.105  0.00  0.00          Si  
ATOM   1143 Si   ROOT    1       8.922  12.221   0.854  0.00  0.00          Si  
ATOM   1144 Si   ROOT    1       7.038  10.866   3.201  0.00  0.00          Si  
ATOM   1145 Si   ROOT    1       4.803   8.879   1.806  0.00  0.00          Si  
ATOM   1146 Si   ROOT    1       2.920   7.523   4.153  0.00  0.00          Si  
ATOM   1147 Si   ROOT    1       3.899   8.607   7.112  0.00  0.00          Si  
ATOM   1148 Si   ROOT    1       7.251   6.769  -5.183  0.00  0.00          Si  
ATOM   1149 Si   ROOT    1       8.230   7.852  -2.224  0.00  0.00          Si  
ATOM   1150 Si   ROOT    1       6.347   6.497   0.123  0.00  0.00          Si  
ATOM   1151 Si   ROOT    1       4.112   4.510  -1.272  0.00  0.00          Si  
ATOM   1152 Si   ROOT    1       2.228   3.155   1.074  0.00  0.00          Si  
ATOM   1153 Si   ROOT    1       3.208   4.238   4.033  0.00  0.00          Si  
ATOM   1154 Si   ROOT    1      11.381   7.006  -1.730  0.00  0.00          Si  
ATOM   1155 Si   ROOT    1      12.361   8.089   1.229  0.00  0.00          Si  
ATOM   1156 Si   ROOT    1      10.478   6.734   3.575  0.00  0.00          Si  
ATOM   1157 Si   ROOT    1       8.242   4.747   2.180  0.00  0.00          Si  
ATOM   1158 Si   ROOT    1       6.359   3.392   4.527  0.00  0.00          Si  
ATOM   1159 Si   ROOT    1       7.339   4.475   7.486  0.00  0.00          Si  
ATOM   1160 Si   ROOT    1      10.690   2.637  -4.809  0.00  0.00          Si  
ATOM   1161 Si   ROOT    1      11.669   3.720  -1.850  0.00  0.00          Si  
ATOM   1162 Si   ROOT    1       9.786   2.365   0.497  0.00  0.00          Si  
ATOM   1163 Si   ROOT    1       7.551   0.378  -0.898  0.00  0.00          Si  
ATOM   1164 Si   ROOT    1       5.667  -0.977   1.449  0.00  0.00          Si  
ATOM   1165 Si   ROOT    1       6.647   0.106   4.408  0.00  0.00          Si  
ATOM   1166 Si   ROOT    1      14.821   2.874  -1.356  0.00  0.00          Si  
ATOM   1167 Si   ROOT    1      15.800   3.957   1.603  0.00  0.00          Si  
ATOM   1168 Si   ROOT    1      13.917   2.602   3.949  0.00  0.00          Si  
ATOM   1169 Si   ROOT    1      11.682   0.615   2.555  0.00  0.00          Si  
ATOM   1170 Si   ROOT    1       9.798  -0.740   4.901  0.00  0.00          Si  
ATOM   1171 Si   ROOT    1      10.778   0.343   7.860  0.00  0.00          Si  
ATOM   1172 Si   ROOT    1      14.129  -1.495  -4.435  0.00  0.00          Si  
ATOM   1173 Si   ROOT    1      15.109  -0.411  -1.475  0.00  0.00          Si  
ATOM   1174 Si   ROOT    1      13.225  -1.767   0.871  0.00  0.00          Si  
ATOM   1175 Si   ROOT    1      10.990  -3.754  -0.524  0.00  0.00          Si  
ATOM   1176 Si   ROOT    1       9.107  -5.109   1.823  0.00  0.00          Si  
ATOM   1177 Si   ROOT    1      10.086  -4.025   4.782  0.00  0.00          Si  
ATOM   1178 Si   ROOT    1      18.260  -1.258  -0.982  0.00  0.00          Si  
ATOM   1179 Si   ROOT    1      19.240  -0.175   1.977  0.00  0.00          Si  
ATOM   1180 Si   ROOT    1      17.356  -1.530   4.324  0.00  0.00          Si  
ATOM   1181 Si   ROOT    1      15.121  -3.517   2.929  0.00  0.00          Si  
ATOM   1182 Si   ROOT    1      13.237  -4.872   5.275  0.00  0.00          Si  
ATOM   1183 Si   ROOT    1      14.217  -3.789   8.234  0.00  0.00          Si  
ATOM   1184 Si   ROOT    1      17.568  -5.627  -4.060  0.00  0.00          Si  
ATOM   1185 Si   ROOT    1      18.548  -4.543  -1.101  0.00  0.00          Si  
ATOM   1186 Si   ROOT    1      16.665  -5.898   1.245  0.00  0.00          Si  
ATOM   1187 Si   ROOT    1      14.429  -7.885  -0.149  0.00  0.00          Si  
ATOM   1188 Si   ROOT    1      12.546  -9.241   2.197  0.00  0.00          Si  
ATOM   1189 Si   ROOT    1      13.525  -8.157   5.156  0.00  0.00          Si  
ATOM   1190 Si   ROOT    1      21.699  -5.390  -0.608  0.00  0.00          Si  
ATOM   1191 Si   ROOT    1      22.679  -4.306   2.351  0.00  0.00          Si  
ATOM   1192 Si   ROOT    1      20.795  -5.661   4.698  0.00  0.00          Si  
ATOM   1193 Si   ROOT    1      18.560  -7.648   3.303  0.00  0.00          Si  
ATOM   1194 Si   ROOT    1      16.677  -9.004   5.650  0.00  0.00          Si  
ATOM   1195 Si   ROOT    1      17.656  -7.920   8.609  0.00  0.00          Si  
ATOM   1196 Si   ROOT    1      21.008  -9.759  -3.686  0.00  0.00          Si  
ATOM   1197 Si   ROOT    1      21.987  -8.675  -0.727  0.00  0.00          Si  
ATOM   1198 Si   ROOT    1      20.104 -10.030   1.620  0.00  0.00          Si  
ATOM   1199 Si   ROOT    1      17.869 -12.017   0.225  0.00  0.00          Si  
ATOM   1200 Si   ROOT    1      15.985 -13.372   2.571  0.00  0.00          Si  
ATOM   1201 Si   ROOT    1      16.965 -12.289   5.530  0.00  0.00          Si  
ATOM   1202 Si   ROOT    1      25.138  -9.522  -0.234  0.00  0.00          Si  
ATOM   1203 Si   ROOT    1      26.118  -8.438   2.726  0.00  0.00          Si  
ATOM   1204 Si   ROOT    1      24.235  -9.793   5.072  0.00  0.00          Si  
ATOM   1205 Si   ROOT    1      21.999 -11.780   3.677  0.00  0.00          Si  
ATOM   1206 Si   ROOT    1      20.116 -13.135   6.024  0.00  0.00          Si  
ATOM   1207 Si   ROOT    1      21.096 -12.052   8.983  0.00  0.00          Si  
ATOM   1208 Si   ROOT    1      24.447 -13.890  -3.312  0.00  0.00          Si  
ATOM   1209 Si   ROOT    1      25.427 -12.807  -0.353  0.00  0.00          Si  
ATOM   1210 Si   ROOT    1      23.543 -14.162   1.994  0.00  0.00          Si  
ATOM   1211 Si   ROOT    1      21.308 -16.149   0.599  0.00  0.00          Si  
ATOM   1212 Si   ROOT    1      19.424 -17.504   2.945  0.00  0.00          Si  
ATOM   1213 Si   ROOT    1      20.404 -16.421   5.905  0.00  0.00          Si  
ATOM   1214 Si   ROOT    1      28.578 -13.653   0.141  0.00  0.00          Si  
ATOM   1215 Si   ROOT    1      29.557 -12.570   3.100  0.00  0.00          Si  
ATOM   1216 Si   ROOT    1      27.674 -13.925   5.446  0.00  0.00          Si  
ATOM   1217 Si   ROOT    1      25.439 -15.912   4.051  0.00  0.00          Si  
ATOM   1218 Si   ROOT    1      23.555 -17.267   6.398  0.00  0.00          Si  
ATOM   1219 Si   ROOT    1      24.535 -16.184   9.357  0.00  0.00          Si  
ATOM   1220 Si   ROOT    1      27.886 -18.022  -2.938  0.00  0.00          Si  
ATOM   1221 Si   ROOT    1      28.866 -16.939   0.022  0.00  0.00          Si  
ATOM   1222 Si   ROOT    1      26.982 -18.294   2.368  0.00  0.00          Si  
ATOM   1223 Si   ROOT    1      24.747 -20.281   0.973  0.00  0.00          Si  
ATOM   1224 Si   ROOT    1      22.864 -21.636   3.320  0.00  0.00          Si  
ATOM   1225 Si   ROOT    1      23.843 -20.552   6.279  0.00  0.00          Si  
ATOM   1226 Si   ROOT    1      32.017 -17.785   0.515  0.00  0.00          Si  
ATOM   1227 Si   ROOT    1      32.997 -16.702   3.474  0.00  0.00          Si  
ATOM   1228 Si   ROOT    1      31.113 -18.057   5.820  0.00  0.00          Si  
ATOM   1229 Si   ROOT    1      28.878 -20.044   4.426  0.00  0.00          Si  
ATOM   1230 Si   ROOT    1      26.994 -21.399   6.772  0.00  0.00          Si  
ATOM   1231 Si   ROOT    1      27.974 -20.316   9.731  0.00  0.00          Si  
ATOM   1232 Si   ROOT    1       1.755  23.770   0.225  0.00  0.00          Si  
ATOM   1233 Si   ROOT    1       2.735  24.853   3.184  0.00  0.00          Si  
ATOM   1234 Si   ROOT    1       0.851  23.498   5.531  0.00  0.00          Si  
ATOM   1235 Si   ROOT    1      -1.384  21.511   4.136  0.00  0.00          Si  
ATOM   1236 Si   ROOT    1      -3.267  20.156   6.483  0.00  0.00          Si  
ATOM   1237 Si   ROOT    1      -2.288  21.239   9.442  0.00  0.00          Si  
ATOM   1238 Si   ROOT    1       5.886  24.006   3.678  0.00  0.00          Si  
ATOM   1239 Si   ROOT    1       6.866  25.090   6.637  0.00  0.00          Si  
ATOM   1240 Si   ROOT    1       4.982  23.735   8.983  0.00  0.00          Si  
ATOM   1241 Si   ROOT    1       2.747  21.748   7.589  0.00  0.00          Si  
ATOM   1242 Si   ROOT    1       0.864  20.393   9.935  0.00  0.00          Si  
ATOM   1243 Si   ROOT    1       1.843  21.476  12.894  0.00  0.00          Si  
ATOM   1244 Si   ROOT    1       5.195  19.638   0.599  0.00  0.00          Si  
ATOM   1245 Si   ROOT    1       6.174  20.721   3.559  0.00  0.00          Si  
ATOM   1246 Si   ROOT    1       4.291  19.366   5.905  0.00  0.00          Si  
ATOM   1247 Si   ROOT    1       2.055  17.379   4.510  0.00  0.00          Si  
ATOM   1248 Si   ROOT    1       0.172  16.024   6.857  0.00  0.00          Si  
ATOM   1249 Si   ROOT    1       1.152  17.107   9.816  0.00  0.00          Si  
ATOM   1250 Si   ROOT    1       9.325  19.875   4.052  0.00  0.00          Si  
ATOM   1251 Si   ROOT    1      10.305  20.958   7.011  0.00  0.00          Si  
ATOM   1252 Si   ROOT    1       8.422  19.603   9.358  0.00  0.00          Si  
ATOM   1253 Si   ROOT    1       6.186  17.616   7.963  0.00  0.00          Si  
ATOM   1254 Si   ROOT    1       4.303  16.261  10.309  0.00  0.00          Si  
ATOM   1255 Si   ROOT    1       5.282  17.344  13.268  0.00  0.00          Si  
ATOM   1256 Si   ROOT    1       8.634  15.506   0.974  0.00  0.00          Si  
ATOM   1257 Si   ROOT    1       9.613  16.590   3.933  0.00  0.00          Si  
ATOM   1258 Si   ROOT    1       7.730  15.234   6.279  0.00  0.00          Si  
ATOM   1259 Si   ROOT    1       5.495  13.247   4.884  0.00  0.00          Si  
ATOM   1260 Si   ROOT    1       3.611  11.892   7.231  0.00  0.00          Si  
ATOM   1261 Si   ROOT    1       4.591  12.976  10.190  0.00  0.00          Si  
ATOM   1262 Si   ROOT    1      12.765  15.743   4.426  0.00  0.00          Si  
ATOM   1263 Si   ROOT    1      13.744  16.826   7.385  0.00  0.00          Si  
ATOM   1264 Si   ROOT    1      11.861  15.471   9.732  0.00  0.00          Si  
ATOM   1265 Si   ROOT    1       9.626  13.484   8.337  0.00  0.00          Si  
ATOM   1266 Si   ROOT    1       7.742  12.129  10.683  0.00  0.00          Si  
ATOM   1267 Si   ROOT    1       8.722  13.212  13.643  0.00  0.00          Si  
ATOM   1268 Si   ROOT    1      12.073  11.374   1.348  0.00  0.00          Si  
ATOM   1269 Si   ROOT    1      13.053  12.458   4.307  0.00  0.00          Si  
ATOM   1270 Si   ROOT    1      11.169  11.102   6.653  0.00  0.00          Si  
ATOM   1271 Si   ROOT    1       8.934   9.116   5.259  0.00  0.00          Si  
ATOM   1272 Si   ROOT    1       7.051   7.760   7.605  0.00  0.00          Si  
ATOM   1273 Si   ROOT    1       8.030   8.844  10.564  0.00  0.00          Si  
ATOM   1274 Si   ROOT    1      16.204  11.611   4.800  0.00  0.00          Si  
ATOM   1275 Si   ROOT    1      17.184  12.695   7.760  0.00  0.00          Si  
ATOM   1276 Si   ROOT    1      15.300  11.339  10.106  0.00  0.00          Si  
ATOM   1277 Si   ROOT    1      13.065   9.352   8.711  0.00  0.00          Si  
ATOM   1278 Si   ROOT    1      11.181   7.997  11.058  0.00  0.00          Si  
ATOM   1279 Si   ROOT    1      12.161   9.081  14.017  0.00  0.00          Si  
ATOM   1280 Si   ROOT    1      15.512   7.243   1.722  0.00  0.00          Si  
ATOM   1281 Si   ROOT    1      16.492   8.326   4.681  0.00  0.00          Si  
ATOM   1282 Si   ROOT    1      14.608   6.971   7.028  0.00  0.00          Si  
ATOM   1283 Si   ROOT    1      12.373   4.984   5.633  0.00  0.00          Si  
ATOM   1284 Si   ROOT    1      10.490   3.629   7.979  0.00  0.00          Si  
ATOM   1285 Si   ROOT    1      11.469   4.712  10.939  0.00  0.00          Si  
ATOM   1286 Si   ROOT    1      19.643   7.479   5.175  0.00  0.00          Si  
ATOM   1287 Si   ROOT    1      20.623   8.563   8.134  0.00  0.00          Si  
ATOM   1288 Si   ROOT    1      18.739   7.208  10.480  0.00  0.00          Si  
ATOM   1289 Si   ROOT    1      16.504   5.221   9.085  0.00  0.00          Si  
ATOM   1290 Si   ROOT    1      14.621   3.865  11.432  0.00  0.00          Si  
ATOM   1291 Si   ROOT    1      15.600   4.949  14.391  0.00  0.00          Si  
ATOM   1292 Si   ROOT    1      18.951   3.111   2.096  0.00  0.00          Si  
ATOM   1293 Si   ROOT    1      19.931   4.194   5.055  0.00  0.00          Si  
ATOM   1294 Si   ROOT    1      18.048   2.839   7.402  0.00  0.00          Si  
ATOM   1295 Si   ROOT    1      15.812   0.852   6.007  0.00  0.00          Si  
ATOM   1296 Si   ROOT    1      13.929  -0.503   8.354  0.00  0.00          Si  
ATOM   1297 Si   ROOT    1      14.909   0.580  11.313  0.00  0.00          Si  
ATOM   1298 Si   ROOT    1      23.082   3.348   5.549  0.00  0.00          Si  
ATOM   1299 Si   ROOT    1      24.062   4.431   8.508  0.00  0.00          Si  
ATOM   1300 Si   ROOT    1      22.179   3.076  10.854  0.00  0.00          Si  
ATOM   1301 Si   ROOT    1      19.943   1.089   9.460  0.00  0.00          Si  
ATOM   1302 Si   ROOT    1      18.060  -0.266  11.806  0.00  0.00          Si  
ATOM   1303 Si   ROOT    1      19.040   0.817  14.765  0.00  0.00          Si  
ATOM   1304 Si   ROOT    1      22.391  -1.021   2.471  0.00  0.00          Si  
ATOM   1305 Si   ROOT    1      23.370   0.062   5.430  0.00  0.00          Si  
ATOM   1306 Si   ROOT    1      21.487  -1.293   7.776  0.00  0.00          Si  
ATOM   1307 Si   ROOT    1      19.252  -3.280   6.381  0.00  0.00          Si  
ATOM   1308 Si   ROOT    1      17.368  -4.635   8.728  0.00  0.00          Si  
ATOM   1309 Si   ROOT    1      18.348  -3.552  11.687  0.00  0.00          Si  
ATOM   1310 Si   ROOT    1      26.522  -0.784   5.923  0.00  0.00          Si  
ATOM   1311 Si   ROOT    1      27.501   0.299   8.882  0.00  0.00          Si  
ATOM   1312 Si   ROOT    1      25.618  -1.056  11.229  0.00  0.00          Si  
ATOM   1313 Si   ROOT    1      23.383  -3.043   9.834  0.00  0.00          Si  
ATOM   1314 Si   ROOT    1      21.499  -4.398  12.180  0.00  0.00          Si  
ATOM   1315 Si   ROOT    1      22.479  -3.315  15.139  0.00  0.00          Si  
ATOM   1316 Si   ROOT    1      25.830  -5.153   2.845  0.00  0.00          Si  
ATOM   1317 Si   ROOT    1      26.810  -4.069   5.804  0.00  0.00          Si  
ATOM   1318 Si   ROOT    1      24.926  -5.425   8.150  0.00  0.00          Si  
ATOM   1319 Si   ROOT    1      22.691  -7.412   6.756  0.00  0.00          Si  
ATOM   1320 Si   ROOT    1      20.808  -8.767   9.102  0.00  0.00          Si  
ATOM   1321 Si   ROOT    1      21.787  -7.683  12.061  0.00  0.00          Si  
ATOM   1322 Si   ROOT    1      29.961  -4.916   6.297  0.00  0.00          Si  
ATOM   1323 Si   ROOT    1      30.941  -3.832   9.256  0.00  0.00          Si  
ATOM   1324 Si   ROOT    1      29.057  -5.188  11.603  0.00  0.00          Si  
ATOM   1325 Si   ROOT    1      26.822  -7.175  10.208  0.00  0.00          Si  
ATOM   1326 Si   ROOT    1      24.938  -8.530  12.555  0.00  0.00          Si  
ATOM   1327 Si   ROOT    1      25.918  -7.446  15.514  0.00  0.00          Si  
ATOM   1328 Si   ROOT    1      29.269  -9.285   3.219  0.00  0.00          Si  
ATOM   1329 Si   ROOT    1      30.249  -8.201   6.178  0.00  0.00          Si  
ATOM   1330 Si   ROOT    1      28.366  -9.556   8.525  0.00  0.00          Si  
ATOM   1331 Si   ROOT    1      26.130 -11.543   7.130  0.00  0.00          Si  
ATOM   1332 Si   ROOT    1      24.247 -12.899   9.476  0.00  0.00          Si  
ATOM   1333 Si   ROOT    1      25.226 -11.815  12.435  0.00  0.00          Si  
ATOM   1334 Si   ROOT    1      33.400  -9.048   6.672  0.00  0.00          Si  
ATOM   1335 Si   ROOT    1      34.380  -7.964   9.631  0.00  0.00          Si  
ATOM   1336 Si   ROOT    1      32.496  -9.319  11.977  0.00  0.00          Si  
ATOM   1337 Si   ROOT    1      30.261 -11.306  10.582  0.00  0.00          Si  
ATOM   1338 Si   ROOT    1      28.378 -12.662  12.929  0.00  0.00          Si  
ATOM   1339 Si   ROOT    1      29.357 -11.578  15.888  0.00  0.00          Si  
ATOM   1340 Si   ROOT    1      32.708 -13.416   3.593  0.00  0.00          Si  
ATOM   1341 Si   ROOT    1      33.688 -12.333   6.552  0.00  0.00          Si  
ATOM   1342 Si   ROOT    1      31.805 -13.688   8.899  0.00  0.00          Si  
ATOM   1343 Si   ROOT    1      29.569 -15.675   7.504  0.00  0.00          Si  
ATOM   1344 Si   ROOT    1      27.686 -17.030   9.851  0.00  0.00          Si  
ATOM   1345 Si   ROOT    1      28.666 -15.947  12.810  0.00  0.00          Si  
ATOM   1346 Si   ROOT    1      36.839 -13.179   7.046  0.00  0.00          Si  
ATOM   1347 Si   ROOT    1      37.819 -12.096  10.005  0.00  0.00          Si  
ATOM   1348 Si   ROOT    1      35.936 -13.451  12.351  0.00  0.00          Si  
ATOM   1349 Si   ROOT    1      33.700 -15.438  10.957  0.00  0.00          Si  
ATOM   1350 Si   ROOT    1      31.817 -16.793  13.303  0.00  0.00          Si  
ATOM   1351 Si   ROOT    1      32.797 -15.710  16.262  0.00  0.00          Si  
ATOM   1352 Si   ROOT    1       6.578  28.375   6.756  0.00  0.00          Si  
ATOM   1353 Si   ROOT    1       7.557  29.459   9.715  0.00  0.00          Si  
ATOM   1354 Si   ROOT    1       5.674  28.104  12.062  0.00  0.00          Si  
ATOM   1355 Si   ROOT    1       3.439  26.117  10.667  0.00  0.00          Si  
ATOM   1356 Si   ROOT    1       1.555  24.761  13.013  0.00  0.00          Si  
ATOM   1357 Si   ROOT    1       2.535  25.845  15.972  0.00  0.00          Si  
ATOM   1358 Si   ROOT    1      10.709  28.612  10.209  0.00  0.00          Si  
ATOM   1359 Si   ROOT    1      11.688  29.696  13.168  0.00  0.00          Si  
ATOM   1360 Si   ROOT    1       9.805  28.340  15.514  0.00  0.00          Si  
ATOM   1361 Si   ROOT    1       7.569  26.353  14.119  0.00  0.00          Si  
ATOM   1362 Si   ROOT    1       5.686  24.998  16.466  0.00  0.00          Si  
ATOM   1363 Si   ROOT    1       6.666  26.082  19.425  0.00  0.00          Si  
ATOM   1364 Si   ROOT    1      10.017  24.243   7.130  0.00  0.00          Si  
ATOM   1365 Si   ROOT    1      10.997  25.327  10.089  0.00  0.00          Si  
ATOM   1366 Si   ROOT    1       9.113  23.972  12.436  0.00  0.00          Si  
ATOM   1367 Si   ROOT    1       6.878  21.985  11.041  0.00  0.00          Si  
ATOM   1368 Si   ROOT    1       4.994  20.630  13.388  0.00  0.00          Si  
ATOM   1369 Si   ROOT    1       5.974  21.713  16.347  0.00  0.00          Si  
ATOM   1370 Si   ROOT    1      14.148  24.480  10.583  0.00  0.00          Si  
ATOM   1371 Si   ROOT    1      15.127  25.564  13.542  0.00  0.00          Si  
ATOM   1372 Si   ROOT    1      13.244  24.209  15.888  0.00  0.00          Si  
ATOM   1373 Si   ROOT    1      11.009  22.222  14.494  0.00  0.00          Si  
ATOM   1374 Si   ROOT    1       9.125  20.866  16.840  0.00  0.00          Si  
ATOM   1375 Si   ROOT    1      10.105  21.950  19.799  0.00  0.00          Si  
ATOM   1376 Si   ROOT    1      13.456  20.112   7.505  0.00  0.00          Si  
ATOM   1377 Si   ROOT    1      14.436  21.195  10.464  0.00  0.00          Si  
ATOM   1378 Si   ROOT    1      12.552  19.840  12.810  0.00  0.00          Si  
ATOM   1379 Si   ROOT    1      10.317  17.853  11.415  0.00  0.00          Si  
ATOM   1380 Si   ROOT    1       8.434  16.498  13.762  0.00  0.00          Si  
ATOM   1381 Si   ROOT    1       9.413  17.581  16.721  0.00  0.00          Si  
ATOM   1382 Si   ROOT    1      17.587  20.349  10.957  0.00  0.00          Si  
ATOM   1383 Si   ROOT    1      18.567  21.432  13.916  0.00  0.00          Si  
ATOM   1384 Si   ROOT    1      16.683  20.077  16.263  0.00  0.00          Si  
ATOM   1385 Si   ROOT    1      14.448  18.090  14.868  0.00  0.00          Si  
ATOM   1386 Si   ROOT    1      12.565  16.735  17.214  0.00  0.00          Si  
ATOM   1387 Si   ROOT    1      13.544  17.818  20.173  0.00  0.00          Si  
ATOM   1388 Si   ROOT    1      16.895  15.980   7.879  0.00  0.00          Si  
ATOM   1389 Si   ROOT    1      17.875  17.063  10.838  0.00  0.00          Si  
ATOM   1390 Si   ROOT    1      15.992  15.708  13.184  0.00  0.00          Si  
ATOM   1391 Si   ROOT    1      13.756  13.721  11.790  0.00  0.00          Si  
ATOM   1392 Si   ROOT    1      11.873  12.366  14.136  0.00  0.00          Si  
ATOM   1393 Si   ROOT    1      12.853  13.449  17.095  0.00  0.00          Si  
ATOM   1394 Si   ROOT    1      21.026  16.217  11.331  0.00  0.00          Si  
ATOM   1395 Si   ROOT    1      22.006  17.300  14.290  0.00  0.00          Si  
ATOM   1396 Si   ROOT    1      20.123  15.945  16.637  0.00  0.00          Si  
ATOM   1397 Si   ROOT    1      17.887  13.958  15.242  0.00  0.00          Si  
ATOM   1398 Si   ROOT    1      16.004  12.603  17.589  0.00  0.00          Si  
ATOM   1399 Si   ROOT    1      16.983  13.686  20.548  0.00  0.00          Si  
ATOM   1400 Si   ROOT    1      20.335  11.848   8.253  0.00  0.00          Si  
ATOM   1401 Si   ROOT    1      21.314  12.932  11.212  0.00  0.00          Si  
ATOM   1402 Si   ROOT    1      19.431  11.576  13.559  0.00  0.00          Si  
ATOM   1403 Si   ROOT    1      17.196   9.589  12.164  0.00  0.00          Si  
ATOM   1404 Si   ROOT    1      15.312   8.234  14.510  0.00  0.00          Si  
ATOM   1405 Si   ROOT    1      16.292   9.318  17.469  0.00  0.00          Si  
ATOM   1406 Si   ROOT    1      24.466  12.085  11.705  0.00  0.00          Si  
ATOM   1407 Si   ROOT    1      25.445  13.169  14.665  0.00  0.00          Si  
ATOM   1408 Si   ROOT    1      23.562  11.813  17.011  0.00  0.00          Si  
ATOM   1409 Si   ROOT    1      21.326   9.826  15.616  0.00  0.00          Si  
ATOM   1410 Si   ROOT    1      19.443   8.471  17.963  0.00  0.00          Si  
ATOM   1411 Si   ROOT    1      20.423   9.555  20.922  0.00  0.00          Si  
ATOM   1412 Si   ROOT    1      23.774   7.716   8.627  0.00  0.00          Si  
ATOM   1413 Si   ROOT    1      24.754   8.800  11.586  0.00  0.00          Si  
ATOM   1414 Si   ROOT    1      22.870   7.445  13.933  0.00  0.00          Si  
ATOM   1415 Si   ROOT    1      20.635   5.458  12.538  0.00  0.00          Si  
ATOM   1416 Si   ROOT    1      18.751   4.102  14.884  0.00  0.00          Si  
ATOM   1417 Si   ROOT    1      19.731   5.186  17.844  0.00  0.00          Si  
ATOM   1418 Si   ROOT    1      27.905   7.953  12.080  0.00  0.00          Si  
ATOM   1419 Si   ROOT    1      28.884   9.037  15.039  0.00  0.00          Si  
ATOM   1420 Si   ROOT    1      27.001   7.682  17.385  0.00  0.00          Si  
ATOM   1421 Si   ROOT    1      24.766   5.695  15.990  0.00  0.00          Si  
ATOM   1422 Si   ROOT    1      22.882   4.339  18.337  0.00  0.00          Si  
ATOM   1423 Si   ROOT    1      23.862   5.423  21.296  0.00  0.00          Si  
ATOM   1424 Si   ROOT    1      27.213   3.585   9.001  0.00  0.00          Si  
ATOM   1425 Si   ROOT    1      28.193   4.668  11.960  0.00  0.00          Si  
ATOM   1426 Si   ROOT    1      26.309   3.313  14.307  0.00  0.00          Si  
ATOM   1427 Si   ROOT    1      24.074   1.326  12.912  0.00  0.00          Si  
ATOM   1428 Si   ROOT    1      22.191  -0.029  15.259  0.00  0.00          Si  
ATOM   1429 Si   ROOT    1      23.170   1.054  18.218  0.00  0.00          Si  
ATOM   1430 Si   ROOT    1      31.344   3.822  12.454  0.00  0.00          Si  
ATOM   1431 Si   ROOT    1      32.324   4.905  15.413  0.00  0.00          Si  
ATOM   1432 Si   ROOT    1      30.440   3.550  17.760  0.00  0.00          Si  
ATOM   1433 Si   ROOT    1      28.205   1.563  16.365  0.00  0.00          Si  
ATOM   1434 Si   ROOT    1      26.322   0.208  18.711  0.00  0.00          Si  
ATOM   1435 Si   ROOT    1      27.301   1.291  21.670  0.00  0.00          Si  
ATOM   1436 Si   ROOT    1      30.652  -0.547   9.376  0.00  0.00          Si  
ATOM   1437 Si   ROOT    1      31.632   0.536  12.335  0.00  0.00          Si  
ATOM   1438 Si   ROOT    1      29.749  -0.819  14.681  0.00  0.00          Si  
ATOM   1439 Si   ROOT    1      27.513  -2.806  13.286  0.00  0.00          Si  
ATOM   1440 Si   ROOT    1      25.630  -4.161  15.633  0.00  0.00          Si  
ATOM   1441 Si   ROOT    1      26.610  -3.078  18.592  0.00  0.00          Si  
ATOM   1442 Si   ROOT    1      34.783  -0.310  12.828  0.00  0.00          Si  
ATOM   1443 Si   ROOT    1      35.763   0.773  15.787  0.00  0.00          Si  
ATOM   1444 Si   ROOT    1      33.880  -0.582  18.134  0.00  0.00          Si  
ATOM   1445 Si   ROOT    1      31.644  -2.569  16.739  0.00  0.00          Si  
ATOM   1446 Si   ROOT    1      29.761  -3.924  19.085  0.00  0.00          Si  
ATOM   1447 Si   ROOT    1      30.740  -2.841  22.045  0.00  0.00          Si  
ATOM   1448 Si   ROOT    1      34.092  -4.679   9.750  0.00  0.00          Si  
ATOM   1449 Si   ROOT    1      35.071  -3.595  12.709  0.00  0.00          Si  
ATOM   1450 Si   ROOT    1      33.188  -4.951  15.055  0.00  0.00          Si  
ATOM   1451 Si   ROOT    1      30.953  -6.938  13.661  0.00  0.00          Si  
ATOM   1452 Si   ROOT    1      29.069  -8.293  16.007  0.00  0.00          Si  
ATOM   1453 Si   ROOT    1      30.049  -7.209  18.966  0.00  0.00          Si  
ATOM   1454 Si   ROOT    1      38.223  -4.442  13.202  0.00  0.00          Si  
ATOM   1455 Si   ROOT    1      39.202  -3.359  16.161  0.00  0.00          Si  
ATOM   1456 Si   ROOT    1      37.319  -4.714  18.508  0.00  0.00          Si  
ATOM   1457 Si   ROOT    1      35.083  -6.701  17.113  0.00  0.00          Si  
ATOM   1458 Si   ROOT    1      33.200  -8.056  19.460  0.00  0.00          Si  
ATOM   1459 Si   ROOT    1      34.180  -6.972  22.419  0.00  0.00          Si  
ATOM   1460 Si   ROOT    1      37.531  -8.811  10.124  0.00  0.00          Si  
ATOM   1461 Si   ROOT    1      38.511  -7.727  13.083  0.00  0.00          Si  
ATOM   1462 Si   ROOT    1      36.627  -9.082  15.430  0.00  0.00          Si  
ATOM   1463 Si   ROOT    1      34.392 -11.069  14.035  0.00  0.00          Si  
ATOM   1464 Si   ROOT    1      32.508 -12.425  16.381  0.00  0.00          Si  
ATOM   1465 Si   ROOT    1      33.488 -11.341  19.340  0.00  0.00          Si  
ATOM   1466 Si   ROOT    1      41.662  -8.574  13.577  0.00  0.00          Si  
ATOM   1467 Si   ROOT    1      42.642  -7.490  16.536  0.00  0.00          Si  
ATOM   1468 Si   ROOT    1      40.758  -8.846  18.882  0.00  0.00          Si  
ATOM   1469 Si   ROOT    1      38.523 -10.832  17.487  0.00  0.00          Si  
ATOM   1470 Si   ROOT    1      36.639 -12.188  19.834  0.00  0.00          Si  
ATOM   1471 Si   ROOT    1      37.619 -11.104  22.793  0.00  0.00          Si  
ATOM   1472 Si   ROOT    1      11.400  32.981  13.287  0.00  0.00          Si  
ATOM   1473 Si   ROOT    1      12.380  34.064  16.246  0.00  0.00          Si  
ATOM   1474 Si   ROOT    1      10.496  32.709  18.592  0.00  0.00          Si  
ATOM   1475 Si   ROOT    1       8.261  30.722  17.198  0.00  0.00          Si  
ATOM   1476 Si   ROOT    1       6.378  29.367  19.544  0.00  0.00          Si  
ATOM   1477 Si   ROOT    1       7.357  30.450  22.503  0.00  0.00          Si  
ATOM   1478 Si   ROOT    1      15.531  33.218  16.739  0.00  0.00          Si  
ATOM   1479 Si   ROOT    1      16.511  34.301  19.698  0.00  0.00          Si  
ATOM   1480 Si   ROOT    1      14.627  32.946  22.045  0.00  0.00          Si  
ATOM   1481 Si   ROOT    1      12.392  30.959  20.650  0.00  0.00          Si  
ATOM   1482 Si   ROOT    1      10.508  29.604  22.997  0.00  0.00          Si  
ATOM   1483 Si   ROOT    1      11.488  30.687  25.956  0.00  0.00          Si  
ATOM   1484 Si   ROOT    1      14.839  28.849  13.661  0.00  0.00          Si  
ATOM   1485 Si   ROOT    1      15.819  29.933  16.620  0.00  0.00          Si  
ATOM   1486 Si   ROOT    1      13.936  28.577  18.967  0.00  0.00          Si  
ATOM   1487 Si   ROOT    1      11.700  26.590  17.572  0.00  0.00          Si  
ATOM   1488 Si   ROOT    1       9.817  25.235  19.918  0.00  0.00          Si  
ATOM   1489 Si   ROOT    1      10.796  26.319  22.877  0.00  0.00          Si  
ATOM   1490 Si   ROOT    1      18.970  29.086  17.114  0.00  0.00          Si  
ATOM   1491 Si   ROOT    1      19.950  30.170  20.073  0.00  0.00          Si  
ATOM   1492 Si   ROOT    1      18.066  28.814  22.419  0.00  0.00          Si  
ATOM   1493 Si   ROOT    1      15.831  26.827  21.024  0.00  0.00          Si  
ATOM   1494 Si   ROOT    1      13.948  25.472  23.371  0.00  0.00          Si  
ATOM   1495 Si   ROOT    1      14.927  26.556  26.330  0.00  0.00          Si  
ATOM   1496 Si   ROOT    1      18.279  24.717  14.035  0.00  0.00          Si  
ATOM   1497 Si   ROOT    1      19.258  25.801  16.994  0.00  0.00          Si  
ATOM   1498 Si   ROOT    1      17.375  24.446  19.341  0.00  0.00          Si  
ATOM   1499 Si   ROOT    1      15.140  22.459  17.946  0.00  0.00          Si  
ATOM   1500 Si   ROOT    1      13.256  21.103  20.293  0.00  0.00          Si  
ATOM   1501 Si   ROOT    1      14.236  22.187  23.252  0.00  0.00          Si  
ATOM   1502 Si   ROOT    1      22.409  24.954  17.488  0.00  0.00          Si  
ATOM   1503 Si   ROOT    1      23.389  26.038  20.447  0.00  0.00          Si  
ATOM   1504 Si   ROOT    1      21.506  24.683  22.793  0.00  0.00          Si  
ATOM   1505 Si   ROOT    1      19.270  22.696  21.399  0.00  0.00          Si  
ATOM   1506 Si   ROOT    1      17.387  21.340  23.745  0.00  0.00          Si  
ATOM   1507 Si   ROOT    1      18.367  22.424  26.704  0.00  0.00          Si  
ATOM   1508 Si   ROOT    1      21.718  20.586  14.410  0.00  0.00          Si  
ATOM   1509 Si   ROOT    1      22.698  21.669  17.369  0.00  0.00          Si  
ATOM   1510 Si   ROOT    1      20.814  20.314  19.715  0.00  0.00          Si  
ATOM   1511 Si   ROOT    1      18.579  18.327  18.320  0.00  0.00          Si  
ATOM   1512 Si   ROOT    1      16.695  16.972  20.667  0.00  0.00          Si  
ATOM   1513 Si   ROOT    1      17.675  18.055  23.626  0.00  0.00          Si  
ATOM   1514 Si   ROOT    1      25.849  20.823  17.862  0.00  0.00          Si  
ATOM   1515 Si   ROOT    1      26.828  21.906  20.821  0.00  0.00          Si  
ATOM   1516 Si   ROOT    1      24.945  20.551  23.168  0.00  0.00          Si  
ATOM   1517 Si   ROOT    1      22.710  18.564  21.773  0.00  0.00          Si  
ATOM   1518 Si   ROOT    1      20.826  17.209  24.119  0.00  0.00          Si  
ATOM   1519 Si   ROOT    1      21.806  18.292  27.078  0.00  0.00          Si  
ATOM   1520 Si   ROOT    1      25.157  16.454  14.784  0.00  0.00          Si  
ATOM   1521 Si   ROOT    1      26.137  17.537  17.743  0.00  0.00          Si  
ATOM   1522 Si   ROOT    1      24.253  16.182  20.089  0.00  0.00          Si  
ATOM   1523 Si   ROOT    1      22.018  14.195  18.695  0.00  0.00          Si  
ATOM   1524 Si   ROOT    1      20.135  12.840  21.041  0.00  0.00          Si  
ATOM   1525 Si   ROOT    1      21.114  13.923  24.000  0.00  0.00          Si  
ATOM   1526 Si   ROOT    1      29.288  16.691  18.236  0.00  0.00          Si  
ATOM   1527 Si   ROOT    1      30.268  17.774  21.195  0.00  0.00          Si  
ATOM   1528 Si   ROOT    1      28.384  16.419  23.542  0.00  0.00          Si  
ATOM   1529 Si   ROOT    1      26.149  14.432  22.147  0.00  0.00          Si  
ATOM   1530 Si   ROOT    1      24.265  13.077  24.494  0.00  0.00          Si  
ATOM   1531 Si   ROOT    1      25.245  14.160  27.453  0.00  0.00          Si  
ATOM   1532 Si   ROOT    1      28.596  12.322  15.158  0.00  0.00          Si  
ATOM   1533 Si   ROOT    1      29.576  13.405  18.117  0.00  0.00          Si  
ATOM   1534 Si   ROOT    1      27.693  12.050  20.464  0.00  0.00          Si  
ATOM   1535 Si   ROOT    1      25.457  10.063  19.069  0.00  0.00          Si  
ATOM   1536 Si   ROOT    1      23.574   8.708  21.415  0.00  0.00          Si  
ATOM   1537 Si   ROOT    1      24.553   9.792  24.374  0.00  0.00          Si  
ATOM   1538 Si   ROOT    1      32.727  12.559  18.610  0.00  0.00          Si  
ATOM   1539 Si   ROOT    1      33.707  13.642  21.570  0.00  0.00          Si  
ATOM   1540 Si   ROOT    1      31.823  12.287  23.916  0.00  0.00          Si  
ATOM   1541 Si   ROOT    1      29.588  10.300  22.521  0.00  0.00          Si  
ATOM   1542 Si   ROOT    1      27.705   8.945  24.868  0.00  0.00          Si  
ATOM   1543 Si   ROOT    1      28.684  10.028  27.827  0.00  0.00          Si  
ATOM   1544 Si   ROOT    1      32.036   8.190  15.532  0.00  0.00          Si  
ATOM   1545 Si   ROOT    1      33.015   9.274  18.491  0.00  0.00          Si  
ATOM   1546 Si   ROOT    1      31.132   7.918  20.838  0.00  0.00          Si  
ATOM   1547 Si   ROOT    1      28.897   5.932  19.443  0.00  0.00          Si  
ATOM   1548 Si   ROOT    1      27.013   4.576  21.789  0.00  0.00          Si  
ATOM   1549 Si   ROOT    1      27.993   5.660  24.749  0.00  0.00          Si  
ATOM   1550 Si   ROOT    1      36.166   8.427  18.985  0.00  0.00          Si  
ATOM   1551 Si   ROOT    1      37.146   9.511  21.944  0.00  0.00          Si  
ATOM   1552 Si   ROOT    1      35.263   8.155  24.290  0.00  0.00          Si  
ATOM   1553 Si   ROOT    1      33.027   6.169  22.896  0.00  0.00          Si  
ATOM   1554 Si   ROOT    1      31.144   4.813  25.242  0.00  0.00          Si  
ATOM   1555 Si   ROOT    1      32.124   5.897  28.201  0.00  0.00          Si  
ATOM   1556 Si   ROOT    1      35.475   4.058  15.906  0.00  0.00          Si  
ATOM   1557 Si   ROOT    1      36.455   5.142  18.866  0.00  0.00          Si  
ATOM   1558 Si   ROOT    1      34.571   3.787  21.212  0.00  0.00          Si  
ATOM   1559 Si   ROOT    1      32.336   1.800  19.817  0.00  0.00          Si  
ATOM   1560 Si   ROOT    1      30.452   0.444  22.164  0.00  0.00          Si  
ATOM   1561 Si   ROOT    1      31.432   1.528  25.123  0.00  0.00          Si  
ATOM   1562 Si   ROOT    1      39.606   4.295  19.359  0.00  0.00          Si  
ATOM   1563 Si   ROOT    1      40.585   5.379  22.318  0.00  0.00          Si  
ATOM   1564 Si   ROOT    1      38.702   4.024  24.665  0.00  0.00          Si  
ATOM   1565 Si   ROOT    1      36.467   2.037  23.270  0.00  0.00          Si  
ATOM   1566 Si   ROOT    1      34.583   0.681  25.616  0.00  0.00          Si  
ATOM   1567 Si   ROOT    1      35.563   1.765  28.575  0.00  0.00          Si  
ATOM   1568 Si   ROOT    1      38.914  -0.073  16.281  0.00  0.00          Si  
ATOM   1569 Si   ROOT    1      39.894   1.010  19.240  0.00  0.00          Si  
ATOM   1570 Si   ROOT    1      38.010  -0.345  21.586  0.00  0.00          Si  
ATOM   1571 Si   ROOT    1      35.775  -2.332  20.191  0.00  0.00          Si  
ATOM   1572 Si   ROOT    1      33.892  -3.687  22.538  0.00  0.00          Si  
ATOM   1573 Si   ROOT    1      34.871  -2.604  25.497  0.00  0.00          Si  
ATOM   1574 Si   ROOT    1      43.045   0.164  19.733  0.00  0.00          Si  
ATOM   1575 Si   ROOT    1      44.025   1.247  22.692  0.00  0.00          Si  
ATOM   1576 Si   ROOT    1      42.141  -0.108  25.039  0.00  0.00          Si  
ATOM   1577 Si   ROOT    1      39.906  -2.095  23.644  0.00  0.00          Si  
ATOM   1578 Si   ROOT    1      38.022  -3.450  25.990  0.00  0.00          Si  
ATOM   1579 Si   ROOT    1      39.002  -2.367  28.950  0.00  0.00          Si  
ATOM   1580 Si   ROOT    1      42.353  -4.205  16.655  0.00  0.00          Si  
ATOM   1581 Si   ROOT    1      43.333  -3.122  19.614  0.00  0.00          Si  
ATOM   1582 Si   ROOT    1      41.450  -4.477  21.960  0.00  0.00          Si  
ATOM   1583 Si   ROOT    1      39.214  -6.464  20.566  0.00  0.00          Si  
ATOM   1584 Si   ROOT    1      37.331  -7.819  22.912  0.00  0.00          Si  
ATOM   1585 Si   ROOT    1      38.310  -6.736  25.871  0.00  0.00          Si  
ATOM   1586 Si   ROOT    1      46.484  -3.968  20.107  0.00  0.00          Si  
ATOM   1587 Si   ROOT    1      47.464  -2.885  23.066  0.00  0.00          Si  
ATOM   1588 Si   ROOT    1      45.580  -4.240  25.413  0.00  0.00          Si  
ATOM   1589 Si   ROOT    1      43.345  -6.227  24.018  0.00  0.00          Si  
ATOM   1590 Si   ROOT    1      41.462  -7.582  26.365  0.00  0.00          Si  
ATOM   1591 Si   ROOT    1      42.441  -6.499  29.324  0.00  0.00          Si  
ATOM   1592 Si   ROOT    1      16.223  37.587  19.818  0.00  0.00          Si  
ATOM   1593 Si   ROOT    1      17.202  38.670  22.777  0.00  0.00          Si  
ATOM   1594 Si   ROOT    1      15.319  37.315  25.123  0.00  0.00          Si  
ATOM   1595 Si   ROOT    1      13.083  35.328  23.728  0.00  0.00          Si  
ATOM   1596 Si   ROOT    1      11.200  33.973  26.075  0.00  0.00          Si  
ATOM   1597 Si   ROOT    1      12.180  35.056  29.034  0.00  0.00          Si  
ATOM   1598 Si   ROOT    1      20.353  37.823  23.270  0.00  0.00          Si  
ATOM   1599 Si   ROOT    1      21.333  38.907  26.229  0.00  0.00          Si  
ATOM   1600 Si   ROOT    1      19.450  37.552  28.576  0.00  0.00          Si  
ATOM   1601 Si   ROOT    1      17.214  35.565  27.181  0.00  0.00          Si  
ATOM   1602 Si   ROOT    1      15.331  34.210  29.528  0.00  0.00          Si  
ATOM   1603 Si   ROOT    1      16.311  35.293  32.487  0.00  0.00          Si  
ATOM   1604 Si   ROOT    1      19.662  33.455  20.192  0.00  0.00          Si  
ATOM   1605 Si   ROOT    1      20.641  34.538  23.151  0.00  0.00          Si  
ATOM   1606 Si   ROOT    1      18.758  33.183  25.497  0.00  0.00          Si  
ATOM   1607 Si   ROOT    1      16.523  31.196  24.103  0.00  0.00          Si  
ATOM   1608 Si   ROOT    1      14.639  29.841  26.449  0.00  0.00          Si  
ATOM   1609 Si   ROOT    1      15.619  30.924  29.408  0.00  0.00          Si  
ATOM   1610 Si   ROOT    1      23.793  33.692  23.644  0.00  0.00          Si  
ATOM   1611 Si   ROOT    1      24.772  34.775  26.604  0.00  0.00          Si  
ATOM   1612 Si   ROOT    1      22.889  33.420  28.950  0.00  0.00          Si  
ATOM   1613 Si   ROOT    1      20.654  31.433  27.555  0.00  0.00          Si  
ATOM   1614 Si   ROOT    1      18.770  30.078  29.902  0.00  0.00          Si  
ATOM   1615 Si   ROOT    1      19.750  31.161  32.861  0.00  0.00          Si  
ATOM   1616 Si   ROOT    1      23.101  29.323  20.566  0.00  0.00          Si  
ATOM   1617 Si   ROOT    1      24.081  30.406  23.525  0.00  0.00          Si  
ATOM   1618 Si   ROOT    1      22.197  29.051  25.872  0.00  0.00          Si  
ATOM   1619 Si   ROOT    1      19.962  27.064  24.477  0.00  0.00          Si  
ATOM   1620 Si   ROOT    1      18.078  25.709  26.823  0.00  0.00          Si  
ATOM   1621 Si   ROOT    1      19.058  26.792  29.782  0.00  0.00          Si  
ATOM   1622 Si   ROOT    1      27.232  29.560  24.019  0.00  0.00          Si  
ATOM   1623 Si   ROOT    1      28.211  30.643  26.978  0.00  0.00          Si  
ATOM   1624 Si   ROOT    1      26.328  29.288  29.324  0.00  0.00          Si  
ATOM   1625 Si   ROOT    1      24.093  27.301  27.929  0.00  0.00          Si  
ATOM   1626 Si   ROOT    1      22.209  25.946  30.276  0.00  0.00          Si  
ATOM   1627 Si   ROOT    1      23.189  27.029  33.235  0.00  0.00          Si  
ATOM   1628 Si   ROOT    1      26.540  25.191  20.940  0.00  0.00          Si  
ATOM   1629 Si   ROOT    1      27.520  26.275  23.899  0.00  0.00          Si  
ATOM   1630 Si   ROOT    1      25.637  24.919  26.246  0.00  0.00          Si  
ATOM   1631 Si   ROOT    1      23.401  22.932  24.851  0.00  0.00          Si  
ATOM   1632 Si   ROOT    1      21.518  21.577  27.198  0.00  0.00          Si  
ATOM   1633 Si   ROOT    1      22.497  22.661  30.157  0.00  0.00          Si  
ATOM   1634 Si   ROOT    1      30.671  25.428  24.393  0.00  0.00          Si  
ATOM   1635 Si   ROOT    1      31.651  26.512  27.352  0.00  0.00          Si  
ATOM   1636 Si   ROOT    1      29.767  25.156  29.698  0.00  0.00          Si  
ATOM   1637 Si   ROOT    1      27.532  23.169  28.304  0.00  0.00          Si  
ATOM   1638 Si   ROOT    1      25.649  21.814  30.650  0.00  0.00          Si  
ATOM   1639 Si   ROOT    1      26.628  22.898  33.609  0.00  0.00          Si  
ATOM   1640 Si   ROOT    1      29.979  21.059  21.315  0.00  0.00          Si  
ATOM   1641 Si   ROOT    1      30.959  22.143  24.274  0.00  0.00          Si  
ATOM   1642 Si   ROOT    1      29.076  20.788  26.620  0.00  0.00          Si  
ATOM   1643 Si   ROOT    1      26.840  18.801  25.225  0.00  0.00          Si  
ATOM   1644 Si   ROOT    1      24.957  17.445  27.572  0.00  0.00          Si  
ATOM   1645 Si   ROOT    1      25.937  18.529  30.531  0.00  0.00          Si  
ATOM   1646 Si   ROOT    1      34.110  21.296  24.767  0.00  0.00          Si  
ATOM   1647 Si   ROOT    1      35.090  22.380  27.726  0.00  0.00          Si  
ATOM   1648 Si   ROOT    1      33.207  21.025  30.073  0.00  0.00          Si  
ATOM   1649 Si   ROOT    1      30.971  19.038  28.678  0.00  0.00          Si  
ATOM   1650 Si   ROOT    1      29.088  17.682  31.024  0.00  0.00          Si  
ATOM   1651 Si   ROOT    1      30.068  18.766  33.983  0.00  0.00          Si  
ATOM   1652 Si   ROOT    1      33.419  16.928  21.689  0.00  0.00          Si  
ATOM   1653 Si   ROOT    1      34.398  18.011  24.648  0.00  0.00          Si  
ATOM   1654 Si   ROOT    1      32.515  16.656  26.994  0.00  0.00          Si  
ATOM   1655 Si   ROOT    1      30.280  14.669  25.600  0.00  0.00          Si  
ATOM   1656 Si   ROOT    1      28.396  13.314  27.946  0.00  0.00          Si  
ATOM   1657 Si   ROOT    1      29.376  14.397  30.905  0.00  0.00          Si  
ATOM   1658 Si   ROOT    1      37.550  17.165  25.141  0.00  0.00          Si  
ATOM   1659 Si   ROOT    1      38.529  18.248  28.100  0.00  0.00          Si  
ATOM   1660 Si   ROOT    1      36.646  16.893  30.447  0.00  0.00          Si  
ATOM   1661 Si   ROOT    1      34.411  14.906  29.052  0.00  0.00          Si  
ATOM   1662 Si   ROOT    1      32.527  13.551  31.399  0.00  0.00          Si  
ATOM   1663 Si   ROOT    1      33.507  14.634  34.358  0.00  0.00          Si  
ATOM   1664 Si   ROOT    1      36.858  12.796  22.063  0.00  0.00          Si  
ATOM   1665 Si   ROOT    1      37.838  13.879  25.022  0.00  0.00          Si  
ATOM   1666 Si   ROOT    1      35.954  12.524  27.369  0.00  0.00          Si  
ATOM   1667 Si   ROOT    1      33.719  10.537  25.974  0.00  0.00          Si  
ATOM   1668 Si   ROOT    1      31.836   9.182  28.320  0.00  0.00          Si  
ATOM   1669 Si   ROOT    1      32.815  10.265  31.279  0.00  0.00          Si  
ATOM   1670 Si   ROOT    1      40.989  13.033  25.516  0.00  0.00          Si  
ATOM   1671 Si   ROOT    1      41.969  14.116  28.475  0.00  0.00          Si  
ATOM   1672 Si   ROOT    1      40.085  12.761  30.821  0.00  0.00          Si  
ATOM   1673 Si   ROOT    1      37.850  10.774  29.426  0.00  0.00          Si  
ATOM   1674 Si   ROOT    1      35.966   9.419  31.773  0.00  0.00          Si  
ATOM   1675 Si   ROOT    1      36.946  10.502  34.732  0.00  0.00          Si  
ATOM   1676 Si   ROOT    1      40.297   8.664  22.437  0.00  0.00          Si  
ATOM   1677 Si   ROOT    1      41.277   9.748  25.396  0.00  0.00          Si  
ATOM   1678 Si   ROOT    1      39.394   8.392  27.743  0.00  0.00          Si  
ATOM   1679 Si   ROOT    1      37.158   6.405  26.348  0.00  0.00          Si  
ATOM   1680 Si   ROOT    1      35.275   5.050  28.694  0.00  0.00          Si  
ATOM   1681 Si   ROOT    1      36.255   6.134  31.654  0.00  0.00          Si  
ATOM   1682 Si   ROOT    1      44.428   8.901  25.890  0.00  0.00          Si  
ATOM   1683 Si   ROOT    1      45.408   9.984  28.849  0.00  0.00          Si  
ATOM   1684 Si   ROOT    1      43.524   8.629  31.195  0.00  0.00          Si  
ATOM   1685 Si   ROOT    1      41.289   6.642  29.801  0.00  0.00          Si  
ATOM   1686 Si   ROOT    1      39.406   5.287  32.147  0.00  0.00          Si  
ATOM   1687 Si   ROOT    1      40.385   6.371  35.106  0.00  0.00          Si  
ATOM   1688 Si   ROOT    1      43.737   4.532  22.811  0.00  0.00          Si  
ATOM   1689 Si   ROOT    1      44.716   5.616  25.771  0.00  0.00          Si  
ATOM   1690 Si   ROOT    1      42.833   4.261  28.117  0.00  0.00          Si  
ATOM   1691 Si   ROOT    1      40.598   2.274  26.722  0.00  0.00          Si  
ATOM   1692 Si   ROOT    1      38.714   0.918  29.069  0.00  0.00          Si  
ATOM   1693 Si   ROOT    1      39.694   2.002  32.028  0.00  0.00          Si  
ATOM   1694 Si   ROOT    1      47.867   4.769  26.264  0.00  0.00          Si  
ATOM   1695 Si   ROOT    1      48.847   5.853  29.223  0.00  0.00          Si  
ATOM   1696 Si   ROOT    1      46.964   4.497  31.570  0.00  0.00          Si  
ATOM   1697 Si   ROOT    1      44.728   2.511  30.175  0.00  0.00          Si  
ATOM   1698 Si   ROOT    1      42.845   1.155  32.521  0.00  0.00          Si  
ATOM   1699 Si   ROOT    1      43.825   2.239  35.480  0.00  0.00          Si  
ATOM   1700 Si   ROOT    1      47.176   0.401  23.186  0.00  0.00          Si  
ATOM   1701 Si   ROOT    1      48.155   1.484  26.145  0.00  0.00          Si  
ATOM   1702 Si   ROOT    1      46.272   0.129  28.491  0.00  0.00          Si  
ATOM   1703 Si   ROOT    1      44.037  -1.858  27.096  0.00  0.00          Si  
ATOM   1704 Si   ROOT    1      42.153  -3.213  29.443  0.00  0.00          Si  
ATOM   1705 Si   ROOT    1      43.133  -2.130  32.402  0.00  0.00          Si  
ATOM   1706 Si   ROOT    1      51.307   0.638  26.638  0.00  0.00          Si  
ATOM   1707 Si   ROOT    1      52.286   1.721  29.597  0.00  0.00          Si  
ATOM   1708 Si   ROOT    1      50.403   0.366  31.944  0.00  0.00          Si  
ATOM   1709 Si   ROOT    1      48.168  -1.621  30.549  0.00  0.00          Si  
ATOM   1710 Si   ROOT    1      46.284  -2.976  32.896  0.00  0.00          Si  
ATOM   1711 Si   ROOT    1      47.264  -1.893  35.855  0.00  0.00          Si  
ATOM   1712 O201 ROOT    1      19.568 -19.147   2.886  0.00  0.00          O201  
ATOM   1713 O202 ROOT    1      17.849 -17.081   2.699  0.00  0.00          O202  
ATOM   1714 O203 ROOT    1      19.914 -16.962   4.425  0.00  0.00          O203  
ATOM   1715 O205 ROOT    1      24.497 -16.590   5.225  0.00  0.00          O205  
ATOM   1716 O206 ROOT    1      29.067 -13.112   1.620  0.00  0.00          O206  
ATOM   1717 O207 ROOT    1      28.722 -15.296   0.081  0.00  0.00          O207  
ATOM   1718 O208 ROOT    1      27.002 -13.230  -0.106  0.00  0.00          O208  
ATOM   1719 O209 ROOT    1      28.616 -13.247   4.273  0.00  0.00          O209  
ATOM   1720 O210 ROOT    1      26.556 -14.919   4.749  0.00  0.00          O210  
ATOM   1721 O211 ROOT    1      26.210 -17.103   3.210  0.00  0.00          O211  
ATOM   1722 O212 ROOT    1      24.491 -15.037   3.023  0.00  0.00          O212  
ATOM   1723 O213 ROOT    1      23.699 -18.910   6.338  0.00  0.00          O213  
ATOM   1724 O214 ROOT    1      21.980 -16.844   6.151  0.00  0.00          O214  
ATOM   1725 O215 ROOT    1      24.045 -16.726   7.877  0.00  0.00          O215  
ATOM   1726 O217 ROOT    1      23.805 -20.958   2.146  0.00  0.00          O217  
ATOM   1727 O218 ROOT    1      28.376 -17.480  -1.458  0.00  0.00          O218  
ATOM   1728 O219 ROOT    1      28.030 -19.665  -2.997  0.00  0.00          O219  
ATOM   1729 O220 ROOT    1      26.310 -17.599  -3.184  0.00  0.00          O220  
ATOM   1730 O221 ROOT    1      27.924 -17.616   1.195  0.00  0.00          O221  
ATOM   1731 O222 ROOT    1      25.865 -19.287   1.671  0.00  0.00          O222  
ATOM   1732 O223 ROOT    1      25.519 -21.472   0.131  0.00  0.00          O223  
ATOM   1733 O224 ROOT    1      23.799 -19.406  -0.056  0.00  0.00          O224  
ATOM   1734 O225 ROOT    1      23.008 -23.279   3.260  0.00  0.00          O225  
ATOM   1735 O226 ROOT    1      21.288 -21.213   3.073  0.00  0.00          O226  
ATOM   1736 O227 ROOT    1      23.353 -21.094   4.799  0.00  0.00          O227  
ATOM   1737 O229 ROOT    1      27.936 -20.721   5.599  0.00  0.00          O229  
ATOM   1738 O230 ROOT    1      32.507 -17.243   1.994  0.00  0.00          O230  
ATOM   1739 O231 ROOT    1      32.161 -19.428   0.455  0.00  0.00          O231  
ATOM   1740 O232 ROOT    1      30.441 -17.362   0.268  0.00  0.00          O232  
ATOM   1741 O233 ROOT    1      32.055 -17.379   4.647  0.00  0.00          O233  
ATOM   1742 O234 ROOT    1      29.996 -19.050   5.123  0.00  0.00          O234  
ATOM   1743 O235 ROOT    1      29.650 -21.235   3.584  0.00  0.00          O235  
ATOM   1744 O236 ROOT    1      27.930 -19.169   3.397  0.00  0.00          O236  
ATOM   1745 O237 ROOT    1      27.138 -23.042   6.713  0.00  0.00          O237  
ATOM   1746 O238 ROOT    1      25.419 -20.976   6.525  0.00  0.00          O238  
ATOM   1747 O239 ROOT    1      27.484 -20.857   8.252  0.00  0.00          O239  
ATOM   1748 O241 ROOT    1      -2.326  20.833   5.309  0.00  0.00          O241  
ATOM   1749 O242 ROOT    1       2.245  24.311   1.705  0.00  0.00          O242  
ATOM   1750 O243 ROOT    1       1.899  22.127   0.166  0.00  0.00          O243  
ATOM   1751 O244 ROOT    1       0.180  24.193  -0.021  0.00  0.00          O244  
ATOM   1752 O245 ROOT    1       1.793  24.175   4.358  0.00  0.00          O245  
ATOM   1753 O246 ROOT    1      -0.266  22.504   4.833  0.00  0.00          O246  
ATOM   1754 O247 ROOT    1      -0.612  20.320   3.294  0.00  0.00          O247  
ATOM   1755 O248 ROOT    1      -2.332  22.386   3.107  0.00  0.00          O248  
ATOM   1756 O249 ROOT    1      -3.123  18.513   6.423  0.00  0.00          O249  
ATOM   1757 O250 ROOT    1      -4.843  20.579   6.236  0.00  0.00          O250  
ATOM   1758 O251 ROOT    1      -2.777  20.697   7.962  0.00  0.00          O251  
ATOM   1759 O253 ROOT    1       1.805  21.070   8.762  0.00  0.00          O253  
ATOM   1760 O254 ROOT    1       6.376  24.548   5.157  0.00  0.00          O254  
ATOM   1761 O255 ROOT    1       6.030  22.364   3.618  0.00  0.00          O255  
ATOM   1762 O256 ROOT    1       4.310  24.430   3.431  0.00  0.00          O256  
ATOM   1763 O257 ROOT    1       5.924  24.412   7.810  0.00  0.00          O257  
ATOM   1764 O258 ROOT    1       3.865  22.741   8.286  0.00  0.00          O258  
ATOM   1765 O259 ROOT    1       3.519  20.557   6.747  0.00  0.00          O259  
ATOM   1766 O260 ROOT    1       1.799  22.623   6.560  0.00  0.00          O260  
ATOM   1767 O261 ROOT    1       1.008  18.750   9.875  0.00  0.00          O261  
ATOM   1768 O262 ROOT    1      -0.712  20.816   9.688  0.00  0.00          O262  
ATOM   1769 O263 ROOT    1       1.353  20.934  11.415  0.00  0.00          O263  
ATOM   1770 O265 ROOT    1       1.114  16.701   5.683  0.00  0.00          O265  
ATOM   1771 O266 ROOT    1       5.684  20.180   2.079  0.00  0.00          O266  
ATOM   1772 O267 ROOT    1       5.339  17.995   0.540  0.00  0.00          O267  
ATOM   1773 O268 ROOT    1       3.619  20.061   0.353  0.00  0.00          O268  
ATOM   1774 O269 ROOT    1       5.232  20.044   4.732  0.00  0.00          O269  
ATOM   1775 O270 ROOT    1       3.173  18.373   5.208  0.00  0.00          O270  
ATOM   1776 O271 ROOT    1       2.827  16.188   3.668  0.00  0.00          O271  
ATOM   1777 O272 ROOT    1       1.108  18.254   3.481  0.00  0.00          O272  
ATOM   1778 O273 ROOT    1       0.316  14.381   6.797  0.00  0.00          O273  
ATOM   1779 O274 ROOT    1      -1.404  16.447   6.610  0.00  0.00          O274  
ATOM   1780 O275 ROOT    1       0.662  16.566   8.336  0.00  0.00          O275  
ATOM   1781 O277 ROOT    1       5.245  16.938   9.136  0.00  0.00          O277  
ATOM   1782 O278 ROOT    1       9.815  20.416   5.531  0.00  0.00          O278  
ATOM   1783 O279 ROOT    1       9.469  18.232   3.992  0.00  0.00          O279  
ATOM   1784 O280 ROOT    1       7.750  20.298   3.805  0.00  0.00          O280  
ATOM   1785 O281 ROOT    1       9.363  20.281   8.184  0.00  0.00          O281  
ATOM   1786 O282 ROOT    1       7.304  18.610   8.660  0.00  0.00          O282  
ATOM   1787 O283 ROOT    1       6.958  16.425   7.121  0.00  0.00          O283  
ATOM   1788 O284 ROOT    1       5.238  18.491   6.934  0.00  0.00          O284  
ATOM   1789 O285 ROOT    1       4.447  14.618  10.250  0.00  0.00          O285  
ATOM   1790 O286 ROOT    1       2.727  16.684  10.063  0.00  0.00          O286  
ATOM   1791 O287 ROOT    1       4.793  16.803  11.789  0.00  0.00          O287  
ATOM   1792 O289 ROOT    1       4.553  12.570   6.058  0.00  0.00          O289  
ATOM   1793 O290 ROOT    1       9.124  16.048   2.453  0.00  0.00          O290  
ATOM   1794 O291 ROOT    1       8.778  13.863   0.914  0.00  0.00          O291  
ATOM   1795 O292 ROOT    1       7.058  15.929   0.727  0.00  0.00          O292  
ATOM   1796 O293 ROOT    1       8.672  15.912   5.106  0.00  0.00          O293  
ATOM   1797 O294 ROOT    1       6.612  14.241   5.582  0.00  0.00          O294  
ATOM   1798 O295 ROOT    1       6.267  12.056   4.043  0.00  0.00          O295  
ATOM   1799 O296 ROOT    1       4.547  14.122   3.856  0.00  0.00          O296  
ATOM   1800 O297 ROOT    1       3.755  10.249   7.171  0.00  0.00          O297  
ATOM   1801 O298 ROOT    1       2.036  12.315   6.984  0.00  0.00          O298  
ATOM   1802 O299 ROOT    1       4.101  12.434   8.710  0.00  0.00          O299  
ATOM   1803 O301 ROOT    1       8.684  12.807   9.510  0.00  0.00          O301  
ATOM   1804 O302 ROOT    1      13.254  16.285   5.906  0.00  0.00          O302  
ATOM   1805 O303 ROOT    1      12.909  14.100   4.367  0.00  0.00          O303  
ATOM   1806 O304 ROOT    1      11.189  16.166   4.179  0.00  0.00          O304  
ATOM   1807 O305 ROOT    1      12.803  16.149   8.559  0.00  0.00          O305  
ATOM   1808 O306 ROOT    1      10.743  14.478   9.034  0.00  0.00          O306  
ATOM   1809 O307 ROOT    1      10.397  12.293   7.495  0.00  0.00          O307  
ATOM   1810 O308 ROOT    1       8.678  14.359   7.308  0.00  0.00          O308  
ATOM   1811 O309 ROOT    1       7.886  10.486  10.624  0.00  0.00          O309  
ATOM   1812 O310 ROOT    1       6.167  12.552  10.437  0.00  0.00          O310  
ATOM   1813 O311 ROOT    1       8.232  12.671  12.163  0.00  0.00          O311  
ATOM   1814 O313 ROOT    1       7.992   8.438   6.432  0.00  0.00          O313  
ATOM   1815 O314 ROOT    1      12.563  11.916   2.827  0.00  0.00          O314  
ATOM   1816 O315 ROOT    1      12.217   9.732   1.288  0.00  0.00          O315  
ATOM   1817 O316 ROOT    1      10.497  11.798   1.101  0.00  0.00          O316  
ATOM   1818 O317 ROOT    1      12.111  11.780   5.480  0.00  0.00          O317  
ATOM   1819 O318 ROOT    1      10.052  10.109   5.956  0.00  0.00          O318  
ATOM   1820 O319 ROOT    1       9.706   7.925   4.417  0.00  0.00          O319  
ATOM   1821 O320 ROOT    1       7.986   9.991   4.230  0.00  0.00          O320  
ATOM   1822 O321 ROOT    1       7.195   6.118   7.546  0.00  0.00          O321  
ATOM   1823 O322 ROOT    1       5.475   8.184   7.358  0.00  0.00          O322  
ATOM   1824 O323 ROOT    1       7.540   8.302   9.085  0.00  0.00          O323  
ATOM   1825 O325 ROOT    1      12.123   8.675   9.884  0.00  0.00          O325  
ATOM   1826 O326 ROOT    1      16.694  12.153   6.280  0.00  0.00          O326  
ATOM   1827 O327 ROOT    1      16.348   9.969   4.741  0.00  0.00          O327  
ATOM   1828 O328 ROOT    1      14.628  12.034   4.554  0.00  0.00          O328  
ATOM   1829 O329 ROOT    1      16.242  12.017   8.933  0.00  0.00          O329  
ATOM   1830 O330 ROOT    1      14.182  10.346   9.409  0.00  0.00          O330  
ATOM   1831 O331 ROOT    1      13.837   8.162   7.869  0.00  0.00          O331  
ATOM   1832 O332 ROOT    1      12.117  10.227   7.682  0.00  0.00          O332  
ATOM   1833 O333 ROOT    1      11.325   6.355  10.998  0.00  0.00          O333  
ATOM   1834 O334 ROOT    1       9.606   8.420  10.811  0.00  0.00          O334  
ATOM   1835 O335 ROOT    1      11.671   8.539  12.537  0.00  0.00          O335  
ATOM   1836 O337 ROOT    1      11.431   4.306   6.806  0.00  0.00          O337  
ATOM   1837 O338 ROOT    1      16.002   7.784   3.202  0.00  0.00          O338  
ATOM   1838 O339 ROOT    1      15.656   5.600   1.663  0.00  0.00          O339  
ATOM   1839 O340 ROOT    1      13.937   7.666   1.475  0.00  0.00          O340  
ATOM   1840 O341 ROOT    1      15.550   7.648   5.854  0.00  0.00          O341  
ATOM   1841 O342 ROOT    1      13.491   5.977   6.330  0.00  0.00          O342  
ATOM   1842 O343 ROOT    1      13.145   3.793   4.791  0.00  0.00          O343  
ATOM   1843 O344 ROOT    1      11.425   5.859   4.604  0.00  0.00          O344  
ATOM   1844 O345 ROOT    1      10.634   1.986   7.920  0.00  0.00          O345  
ATOM   1845 O346 ROOT    1       8.914   4.052   7.733  0.00  0.00          O346  
ATOM   1846 O347 ROOT    1      10.980   4.170   9.459  0.00  0.00          O347  
ATOM   1847 O349 ROOT    1      15.562   4.543  10.259  0.00  0.00          O349  
ATOM   1848 O350 ROOT    1      20.133   8.021   6.654  0.00  0.00          O350  
ATOM   1849 O351 ROOT    1      19.787   5.837   5.115  0.00  0.00          O351  
ATOM   1850 O352 ROOT    1      18.068   7.903   4.928  0.00  0.00          O352  
ATOM   1851 O353 ROOT    1      19.681   7.885   9.307  0.00  0.00          O353  
ATOM   1852 O354 ROOT    1      17.622   6.214   9.783  0.00  0.00          O354  
ATOM   1853 O355 ROOT    1      17.276   4.030   8.244  0.00  0.00          O355  
ATOM   1854 O356 ROOT    1      15.556   6.096   8.057  0.00  0.00          O356  
ATOM   1855 O357 ROOT    1      14.765   2.223  11.372  0.00  0.00          O357  
ATOM   1856 O358 ROOT    1      13.045   4.289  11.185  0.00  0.00          O358  
ATOM   1857 O359 ROOT    1      15.110   4.407  12.911  0.00  0.00          O359  
ATOM   1858 O361 ROOT    1      14.871   0.174   7.180  0.00  0.00          O361  
ATOM   1859 O362 ROOT    1      19.441   3.652   3.576  0.00  0.00          O362  
ATOM   1860 O363 ROOT    1      19.096   1.468   2.037  0.00  0.00          O363  
ATOM   1861 O364 ROOT    1      17.376   3.534   1.850  0.00  0.00          O364  
ATOM   1862 O365 ROOT    1      18.989   3.517   6.229  0.00  0.00          O365  
ATOM   1863 O366 ROOT    1      16.930   1.845   6.704  0.00  0.00          O366  
ATOM   1864 O367 ROOT    1      16.584  -0.339   5.165  0.00  0.00          O367  
ATOM   1865 O368 ROOT    1      14.865   1.727   4.978  0.00  0.00          O368  
ATOM   1866 O369 ROOT    1      14.073  -2.146   8.294  0.00  0.00          O369  
ATOM   1867 O370 ROOT    1      12.353  -0.080   8.107  0.00  0.00          O370  
ATOM   1868 O371 ROOT    1      14.419   0.038   9.833  0.00  0.00          O371  
ATOM   1869 O373 ROOT    1      19.002   0.411  10.633  0.00  0.00          O373  
ATOM   1870 O374 ROOT    1      23.572   3.889   7.028  0.00  0.00          O374  
ATOM   1871 O375 ROOT    1      23.226   1.705   5.489  0.00  0.00          O375  
ATOM   1872 O376 ROOT    1      21.507   3.771   5.302  0.00  0.00          O376  
ATOM   1873 O377 ROOT    1      23.120   3.753   9.681  0.00  0.00          O377  
ATOM   1874 O378 ROOT    1      21.061   2.082  10.157  0.00  0.00          O378  
ATOM   1875 O379 ROOT    1      20.715  -0.102   8.618  0.00  0.00          O379  
ATOM   1876 O380 ROOT    1      18.996   1.964   8.431  0.00  0.00          O380  
ATOM   1877 O381 ROOT    1      18.204  -1.909  11.747  0.00  0.00          O381  
ATOM   1878 O382 ROOT    1      16.484   0.157  11.559  0.00  0.00          O382  
ATOM   1879 O383 ROOT    1      18.550   0.275  13.286  0.00  0.00          O383  
ATOM   1880 O385 ROOT    1      18.310  -3.957   7.555  0.00  0.00          O385  
ATOM   1881 O386 ROOT    1      22.881  -0.479   3.950  0.00  0.00          O386  
ATOM   1882 O387 ROOT    1      22.535  -2.664   2.411  0.00  0.00          O387  
ATOM   1883 O388 ROOT    1      20.815  -0.598   2.224  0.00  0.00          O388  
ATOM   1884 O389 ROOT    1      22.429  -0.615   6.603  0.00  0.00          O389  
ATOM   1885 O390 ROOT    1      20.369  -2.286   7.079  0.00  0.00          O390  
ATOM   1886 O391 ROOT    1      20.024  -4.471   5.540  0.00  0.00          O391  
ATOM   1887 O392 ROOT    1      18.304  -2.405   5.352  0.00  0.00          O392  
ATOM   1888 O393 ROOT    1      17.512  -6.278   8.668  0.00  0.00          O393  
ATOM   1889 O394 ROOT    1      15.793  -4.212   8.481  0.00  0.00          O394  
ATOM   1890 O395 ROOT    1      17.858  -4.093  10.207  0.00  0.00          O395  
ATOM   1891 O397 ROOT    1      22.441  -3.720  11.007  0.00  0.00          O397  
ATOM   1892 O398 ROOT    1      27.011  -0.242   7.403  0.00  0.00          O398  
ATOM   1893 O399 ROOT    1      26.666  -2.427   5.863  0.00  0.00          O399  
ATOM   1894 O400 ROOT    1      24.946  -0.361   5.676  0.00  0.00          O400  
ATOM   1895 O401 ROOT    1      26.560  -0.378  10.055  0.00  0.00          O401  
ATOM   1896 O402 ROOT    1      24.500  -2.049  10.531  0.00  0.00          O402  
ATOM   1897 O403 ROOT    1      24.154  -4.234   8.992  0.00  0.00          O403  
ATOM   1898 O404 ROOT    1      22.435  -2.168   8.805  0.00  0.00          O404  
ATOM   1899 O405 ROOT    1      21.643  -6.041  12.121  0.00  0.00          O405  
ATOM   1900 O406 ROOT    1      19.923  -3.975  11.934  0.00  0.00          O406  
ATOM   1901 O407 ROOT    1      21.989  -3.856  13.660  0.00  0.00          O407  
ATOM   1902 O409 ROOT    1      21.749  -8.089   7.929  0.00  0.00          O409  
ATOM   1903 O410 ROOT    1      26.320  -4.611   4.324  0.00  0.00          O410  
ATOM   1904 O411 ROOT    1      25.974  -6.795   2.785  0.00  0.00          O411  
ATOM   1905 O412 ROOT    1      24.254  -4.730   2.598  0.00  0.00          O412  
ATOM   1906 O413 ROOT    1      25.868  -4.747   6.977  0.00  0.00          O413  
ATOM   1907 O414 ROOT    1      23.809  -6.418   7.453  0.00  0.00          O414  
ATOM   1908 O415 ROOT    1      23.463  -8.602   5.914  0.00  0.00          O415  
ATOM   1909 O416 ROOT    1      21.743  -6.537   5.727  0.00  0.00          O416  
ATOM   1910 O417 ROOT    1      20.952 -10.409   9.042  0.00  0.00          O417  
ATOM   1911 O418 ROOT    1      19.232  -8.344   8.855  0.00  0.00          O418  
ATOM   1912 O419 ROOT    1      21.297  -8.225  10.582  0.00  0.00          O419  
ATOM   1913 O421 ROOT    1      25.880  -7.852  11.381  0.00  0.00          O421  
ATOM   1914 O422 ROOT    1      30.451  -4.374   7.777  0.00  0.00          O422  
ATOM   1915 O423 ROOT    1      30.105  -6.558   6.238  0.00  0.00          O423  
ATOM   1916 O424 ROOT    1      28.385  -4.493   6.051  0.00  0.00          O424  
ATOM   1917 O425 ROOT    1      29.999  -4.510  10.430  0.00  0.00          O425  
ATOM   1918 O426 ROOT    1      27.939  -6.181  10.905  0.00  0.00          O426  
ATOM   1919 O427 ROOT    1      27.594  -8.365   9.366  0.00  0.00          O427  
ATOM   1920 O428 ROOT    1      25.874  -6.300   9.179  0.00  0.00          O428  
ATOM   1921 O429 ROOT    1      25.082 -10.172  12.495  0.00  0.00          O429  
ATOM   1922 O430 ROOT    1      23.363  -8.107  12.308  0.00  0.00          O430  
ATOM   1923 O431 ROOT    1      25.428  -7.988  14.034  0.00  0.00          O431  
ATOM   1924 O433 ROOT    1      25.189 -12.221   8.303  0.00  0.00          O433  
ATOM   1925 O434 ROOT    1      29.759  -8.743   4.699  0.00  0.00          O434  
ATOM   1926 O435 ROOT    1      29.413 -10.927   3.159  0.00  0.00          O435  
ATOM   1927 O436 ROOT    1      27.694  -8.861   2.972  0.00  0.00          O436  
ATOM   1928 O437 ROOT    1      29.307  -8.879   7.351  0.00  0.00          O437  
ATOM   1929 O438 ROOT    1      27.248 -10.550   7.827  0.00  0.00          O438  
ATOM   1930 O439 ROOT    1      26.902 -12.734   6.288  0.00  0.00          O439  
ATOM   1931 O440 ROOT    1      25.182 -10.668   6.101  0.00  0.00          O440  
ATOM   1932 O441 ROOT    1      24.391 -14.541   9.417  0.00  0.00          O441  
ATOM   1933 O442 ROOT    1      22.671 -12.475   9.230  0.00  0.00          O442  
ATOM   1934 O443 ROOT    1      24.737 -12.357  10.956  0.00  0.00          O443  
ATOM   1935 O445 ROOT    1      29.319 -11.984  11.756  0.00  0.00          O445  
ATOM   1936 O446 ROOT    1      33.890  -8.506   8.151  0.00  0.00          O446  
ATOM   1937 O447 ROOT    1      33.544 -10.690   6.612  0.00  0.00          O447  
ATOM   1938 O448 ROOT    1      31.825  -8.624   6.425  0.00  0.00          O448  
ATOM   1939 O449 ROOT    1      33.438  -8.642  10.804  0.00  0.00          O449  
ATOM   1940 O450 ROOT    1      31.379 -10.313  11.280  0.00  0.00          O450  
ATOM   1941 O451 ROOT    1      31.033 -12.497   9.741  0.00  0.00          O451  
ATOM   1942 O452 ROOT    1      29.313 -10.431   9.553  0.00  0.00          O452  
ATOM   1943 O453 ROOT    1      28.522 -14.304  12.869  0.00  0.00          O453  
ATOM   1944 O454 ROOT    1      26.802 -12.238  12.682  0.00  0.00          O454  
ATOM   1945 O455 ROOT    1      28.867 -12.120  14.408  0.00  0.00          O455  
ATOM   1946 O457 ROOT    1      28.628 -16.353   8.677  0.00  0.00          O457  
ATOM   1947 O458 ROOT    1      33.198 -12.875   5.073  0.00  0.00          O458  
ATOM   1948 O459 ROOT    1      32.853 -15.059   3.534  0.00  0.00          O459  
ATOM   1949 O460 ROOT    1      31.133 -12.993   3.346  0.00  0.00          O460  
ATOM   1950 O461 ROOT    1      32.746 -13.010   7.726  0.00  0.00          O461  
ATOM   1951 O462 ROOT    1      30.687 -14.682   8.201  0.00  0.00          O462  
ATOM   1952 O463 ROOT    1      30.341 -16.866   6.662  0.00  0.00          O463  
ATOM   1953 O464 ROOT    1      28.622 -14.800   6.475  0.00  0.00          O464  
ATOM   1954 O465 ROOT    1      27.830 -18.673   9.791  0.00  0.00          O465  
ATOM   1955 O466 ROOT    1      26.110 -16.607   9.604  0.00  0.00          O466  
ATOM   1956 O467 ROOT    1      28.176 -16.489  11.330  0.00  0.00          O467  
ATOM   1957 O469 ROOT    1      32.759 -16.116  12.130  0.00  0.00          O469  
ATOM   1958 O470 ROOT    1      37.329 -12.638   8.525  0.00  0.00          O470  
ATOM   1959 O471 ROOT    1      36.983 -14.822   6.986  0.00  0.00          O471  
ATOM   1960 O472 ROOT    1      35.264 -12.756   6.799  0.00  0.00          O472  
ATOM   1961 O473 ROOT    1      36.877 -12.774  11.178  0.00  0.00          O473  
ATOM   1962 O474 ROOT    1      34.818 -14.445  11.654  0.00  0.00          O474  
ATOM   1963 O475 ROOT    1      34.472 -16.629  10.115  0.00  0.00          O475  
ATOM   1964 O476 ROOT    1      32.753 -14.563   9.928  0.00  0.00          O476  
ATOM   1965 O477 ROOT    1      31.961 -18.436  13.243  0.00  0.00          O477  
ATOM   1966 O478 ROOT    1      30.241 -16.370  13.056  0.00  0.00          O478  
ATOM   1967 O479 ROOT    1      32.307 -16.252  14.783  0.00  0.00          O479  
ATOM   1968 O481 ROOT    1       2.497  25.439  11.840  0.00  0.00          O481  
ATOM   1969 O482 ROOT    1       7.067  28.917   8.236  0.00  0.00          O482  
ATOM   1970 O483 ROOT    1       6.722  26.733   6.696  0.00  0.00          O483  
ATOM   1971 O484 ROOT    1       5.002  28.798   6.509  0.00  0.00          O484  
ATOM   1972 O485 ROOT    1       6.616  28.781  10.888  0.00  0.00          O485  
ATOM   1973 O486 ROOT    1       4.556  27.110  11.364  0.00  0.00          O486  
ATOM   1974 O487 ROOT    1       4.210  24.926   9.825  0.00  0.00          O487  
ATOM   1975 O488 ROOT    1       2.491  26.992   9.638  0.00  0.00          O488  
ATOM   1976 O489 ROOT    1       1.699  23.119  12.954  0.00  0.00          O489  
ATOM   1977 O490 ROOT    1      -0.020  25.185  12.767  0.00  0.00          O490  
ATOM   1978 O491 ROOT    1       2.045  25.303  14.493  0.00  0.00          O491  
ATOM   1979 O493 ROOT    1       6.628  25.676  15.293  0.00  0.00          O493  
ATOM   1980 O494 ROOT    1      11.198  29.154  11.688  0.00  0.00          O494  
ATOM   1981 O495 ROOT    1      10.853  26.970  10.149  0.00  0.00          O495  
ATOM   1982 O496 ROOT    1       9.133  29.035   9.962  0.00  0.00          O496  
ATOM   1983 O497 ROOT    1      10.747  29.018  14.341  0.00  0.00          O497  
ATOM   1984 O498 ROOT    1       8.687  27.347  14.817  0.00  0.00          O498  
ATOM   1985 O499 ROOT    1       8.341  25.163  13.278  0.00  0.00          O499  
ATOM   1986 O500 ROOT    1       6.622  27.228  13.091  0.00  0.00          O500  
ATOM   1987 O501 ROOT    1       5.830  23.356  16.406  0.00  0.00          O501  
ATOM   1988 O502 ROOT    1       4.110  25.422  16.219  0.00  0.00          O502  
ATOM   1989 O503 ROOT    1       6.176  25.540  17.945  0.00  0.00          O503  
ATOM   1990 O505 ROOT    1       5.936  21.307  12.214  0.00  0.00          O505  
ATOM   1991 O506 ROOT    1      10.507  24.785   8.610  0.00  0.00          O506  
ATOM   1992 O507 ROOT    1      10.161  22.601   7.071  0.00  0.00          O507  
ATOM   1993 O508 ROOT    1       8.441  24.667   6.884  0.00  0.00          O508  
ATOM   1994 O509 ROOT    1      10.055  24.649  11.263  0.00  0.00          O509  
ATOM   1995 O510 ROOT    1       7.996  22.978  11.738  0.00  0.00          O510  
ATOM   1996 O511 ROOT    1       7.650  20.794  10.199  0.00  0.00          O511  
ATOM   1997 O512 ROOT    1       5.930  22.860  10.012  0.00  0.00          O512  
ATOM   1998 O513 ROOT    1       5.138  18.987  13.328  0.00  0.00          O513  
ATOM   1999 O514 ROOT    1       3.419  21.053  13.141  0.00  0.00          O514  
ATOM   2000 O515 ROOT    1       5.484  21.171  14.867  0.00  0.00          O515  
ATOM   2001 O517 ROOT    1      10.067  21.544  15.667  0.00  0.00          O517  
ATOM   2002 O518 ROOT    1      14.638  25.022  12.062  0.00  0.00          O518  
ATOM   2003 O519 ROOT    1      14.292  22.838  10.523  0.00  0.00          O519  
ATOM   2004 O520 ROOT    1      12.572  24.904  10.336  0.00  0.00          O520  
ATOM   2005 O521 ROOT    1      14.186  24.886  14.715  0.00  0.00          O521  
ATOM   2006 O522 ROOT    1      12.126  23.215  15.191  0.00  0.00          O522  
ATOM   2007 O523 ROOT    1      11.781  21.031  13.652  0.00  0.00          O523  
ATOM   2008 O524 ROOT    1      10.061  23.097  13.465  0.00  0.00          O524  
ATOM   2009 O525 ROOT    1       9.269  19.224  16.781  0.00  0.00          O525  
ATOM   2010 O526 ROOT    1       7.550  21.290  16.593  0.00  0.00          O526  
ATOM   2011 O527 ROOT    1       9.615  21.408  18.320  0.00  0.00          O527  
ATOM   2012 O529 ROOT    1       9.375  17.175  12.589  0.00  0.00          O529  
ATOM   2013 O530 ROOT    1      13.946  20.653   8.984  0.00  0.00          O530  
ATOM   2014 O531 ROOT    1      13.600  18.469   7.445  0.00  0.00          O531  
ATOM   2015 O532 ROOT    1      11.881  20.535   7.258  0.00  0.00          O532  
ATOM   2016 O533 ROOT    1      13.494  20.518  11.637  0.00  0.00          O533  
ATOM   2017 O534 ROOT    1      11.435  18.846  12.113  0.00  0.00          O534  
ATOM   2018 O535 ROOT    1      11.089  16.662  10.574  0.00  0.00          O535  
ATOM   2019 O536 ROOT    1       9.369  18.728  10.386  0.00  0.00          O536  
ATOM   2020 O537 ROOT    1       8.578  14.855  13.702  0.00  0.00          O537  
ATOM   2021 O538 ROOT    1       6.858  16.921  13.515  0.00  0.00          O538  
ATOM   2022 O539 ROOT    1       8.924  17.039  15.241  0.00  0.00          O539  
ATOM   2023 O541 ROOT    1      13.506  17.412  16.041  0.00  0.00          O541  
ATOM   2024 O542 ROOT    1      18.077  20.890  12.437  0.00  0.00          O542  
ATOM   2025 O543 ROOT    1      17.731  18.706  10.897  0.00  0.00          O543  
ATOM   2026 O544 ROOT    1      16.011  20.772  10.710  0.00  0.00          O544  
ATOM   2027 O545 ROOT    1      17.625  20.755  15.089  0.00  0.00          O545  
ATOM   2028 O546 ROOT    1      15.566  19.083  15.565  0.00  0.00          O546  
ATOM   2029 O547 ROOT    1      15.220  16.899  14.026  0.00  0.00          O547  
ATOM   2030 O548 ROOT    1      13.500  18.965  13.839  0.00  0.00          O548  
ATOM   2031 O549 ROOT    1      12.709  15.092  17.155  0.00  0.00          O549  
ATOM   2032 O550 ROOT    1      10.989  17.158  16.968  0.00  0.00          O550  
ATOM   2033 O551 ROOT    1      13.054  17.276  18.694  0.00  0.00          O551  
ATOM   2034 O553 ROOT    1      12.815  13.044  12.963  0.00  0.00          O553  
ATOM   2035 O554 ROOT    1      17.385  16.522   9.358  0.00  0.00          O554  
ATOM   2036 O555 ROOT    1      17.040  14.337   7.819  0.00  0.00          O555  
ATOM   2037 O556 ROOT    1      15.320  16.403   7.632  0.00  0.00          O556  
ATOM   2038 O557 ROOT    1      16.933  16.386  12.011  0.00  0.00          O557  
ATOM   2039 O558 ROOT    1      14.874  14.715  12.487  0.00  0.00          O558  
ATOM   2040 O559 ROOT    1      14.528  12.530  10.948  0.00  0.00          O559  
ATOM   2041 O560 ROOT    1      12.809  14.596  10.761  0.00  0.00          O560  
ATOM   2042 O561 ROOT    1      12.017  10.723  14.076  0.00  0.00          O561  
ATOM   2043 O562 ROOT    1      10.297  12.789  13.889  0.00  0.00          O562  
ATOM   2044 O563 ROOT    1      12.363  12.908  15.616  0.00  0.00          O563  
ATOM   2045 O565 ROOT    1      16.945  13.281  16.415  0.00  0.00          O565  
ATOM   2046 O566 ROOT    1      21.516  16.759  12.811  0.00  0.00          O566  
ATOM   2047 O567 ROOT    1      21.170  14.574  11.272  0.00  0.00          O567  
ATOM   2048 O568 ROOT    1      19.451  16.640  11.085  0.00  0.00          O568  
ATOM   2049 O569 ROOT    1      21.064  16.623  15.464  0.00  0.00          O569  
ATOM   2050 O570 ROOT    1      19.005  14.952  15.939  0.00  0.00          O570  
ATOM   2051 O571 ROOT    1      18.659  12.767  14.400  0.00  0.00          O571  
ATOM   2052 O572 ROOT    1      16.939  14.833  14.213  0.00  0.00          O572  
ATOM   2053 O573 ROOT    1      16.148  10.960  17.529  0.00  0.00          O573  
ATOM   2054 O574 ROOT    1      14.428  13.026  17.342  0.00  0.00          O574  
ATOM   2055 O575 ROOT    1      16.494  13.145  19.068  0.00  0.00          O575  
ATOM   2056 O577 ROOT    1      16.254   8.912  13.337  0.00  0.00          O577  
ATOM   2057 O578 ROOT    1      20.824  12.390   9.732  0.00  0.00          O578  
ATOM   2058 O579 ROOT    1      20.479  10.205   8.193  0.00  0.00          O579  
ATOM   2059 O580 ROOT    1      18.759  12.271   8.006  0.00  0.00          O580  
ATOM   2060 O581 ROOT    1      20.373  12.254  12.385  0.00  0.00          O581  
ATOM   2061 O582 ROOT    1      18.313  10.583  12.861  0.00  0.00          O582  
ATOM   2062 O583 ROOT    1      17.968   8.399  11.322  0.00  0.00          O583  
ATOM   2063 O584 ROOT    1      16.248  10.464  11.135  0.00  0.00          O584  
ATOM   2064 O585 ROOT    1      15.456   6.592  14.451  0.00  0.00          O585  
ATOM   2065 O586 ROOT    1      13.737   8.657  14.264  0.00  0.00          O586  
ATOM   2066 O587 ROOT    1      15.802   8.776  15.990  0.00  0.00          O587  
ATOM   2067 O589 ROOT    1      20.385   9.149  16.790  0.00  0.00          O589  
ATOM   2068 O590 ROOT    1      24.955  12.627  13.185  0.00  0.00          O590  
ATOM   2069 O591 ROOT    1      24.610  10.442  11.646  0.00  0.00          O591  
ATOM   2070 O592 ROOT    1      22.890  12.508  11.459  0.00  0.00          O592  
ATOM   2071 O593 ROOT    1      24.504  12.491  15.838  0.00  0.00          O593  
ATOM   2072 O594 ROOT    1      22.444  10.820  16.314  0.00  0.00          O594  
ATOM   2073 O595 ROOT    1      22.098   8.635  14.775  0.00  0.00          O595  
ATOM   2074 O596 ROOT    1      20.379  10.701  14.587  0.00  0.00          O596  
ATOM   2075 O597 ROOT    1      19.587   6.829  17.903  0.00  0.00          O597  
ATOM   2076 O598 ROOT    1      17.867   8.894  17.716  0.00  0.00          O598  
ATOM   2077 O599 ROOT    1      19.933   9.013  19.442  0.00  0.00          O599  
ATOM   2078 O601 ROOT    1      19.693   4.780  13.711  0.00  0.00          O601  
ATOM   2079 O602 ROOT    1      24.264   8.258  10.107  0.00  0.00          O602  
ATOM   2080 O603 ROOT    1      23.918   6.074   8.568  0.00  0.00          O603  
ATOM   2081 O604 ROOT    1      22.198   8.140   8.380  0.00  0.00          O604  
ATOM   2082 O605 ROOT    1      23.812   8.122  12.760  0.00  0.00          O605  
ATOM   2083 O606 ROOT    1      21.752   6.451  13.235  0.00  0.00          O606  
ATOM   2084 O607 ROOT    1      21.407   4.267  11.696  0.00  0.00          O607  
ATOM   2085 O608 ROOT    1      19.687   6.333  11.509  0.00  0.00          O608  
ATOM   2086 O609 ROOT    1      18.895   2.460  14.825  0.00  0.00          O609  
ATOM   2087 O610 ROOT    1      17.176   4.526  14.638  0.00  0.00          O610  
ATOM   2088 O611 ROOT    1      19.241   4.644  16.364  0.00  0.00          O611  
ATOM   2089 O613 ROOT    1      23.824   5.017  17.164  0.00  0.00          O613  
ATOM   2090 O614 ROOT    1      28.395   8.495  13.559  0.00  0.00          O614  
ATOM   2091 O615 ROOT    1      28.049   6.311  12.020  0.00  0.00          O615  
ATOM   2092 O616 ROOT    1      26.329   8.377  11.833  0.00  0.00          O616  
ATOM   2093 O617 ROOT    1      27.943   8.359  16.212  0.00  0.00          O617  
ATOM   2094 O618 ROOT    1      25.883   6.688  16.688  0.00  0.00          O618  
ATOM   2095 O619 ROOT    1      25.538   4.504  15.149  0.00  0.00          O619  
ATOM   2096 O620 ROOT    1      23.818   6.570  14.962  0.00  0.00          O620  
ATOM   2097 O621 ROOT    1      23.026   2.697  18.277  0.00  0.00          O621  
ATOM   2098 O622 ROOT    1      21.307   4.763  18.090  0.00  0.00          O622  
ATOM   2099 O623 ROOT    1      23.372   4.881  19.817  0.00  0.00          O623  
ATOM   2100 O625 ROOT    1      23.132   0.648  14.085  0.00  0.00          O625  
ATOM   2101 O626 ROOT    1      27.703   4.126  10.481  0.00  0.00          O626  
ATOM   2102 O627 ROOT    1      27.357   1.942   8.942  0.00  0.00          O627  
ATOM   2103 O628 ROOT    1      25.638   4.008   8.755  0.00  0.00          O628  
ATOM   2104 O629 ROOT    1      27.251   3.990  13.134  0.00  0.00          O629  
ATOM   2105 O630 ROOT    1      25.192   2.319  13.610  0.00  0.00          O630  
ATOM   2106 O631 ROOT    1      24.846   0.135  12.070  0.00  0.00          O631  
ATOM   2107 O632 ROOT    1      23.126   2.201  11.883  0.00  0.00          O632  
ATOM   2108 O633 ROOT    1      22.335  -1.672  15.199  0.00  0.00          O633  
ATOM   2109 O634 ROOT    1      20.615   0.394  15.012  0.00  0.00          O634  
ATOM   2110 O635 ROOT    1      22.680   0.512  16.738  0.00  0.00          O635  
ATOM   2111 O637 ROOT    1      27.263   0.885  17.538  0.00  0.00          O637  
ATOM   2112 O638 ROOT    1      31.834   4.363  13.933  0.00  0.00          O638  
ATOM   2113 O639 ROOT    1      31.488   2.179  12.394  0.00  0.00          O639  
ATOM   2114 O640 ROOT    1      29.768   4.245  12.207  0.00  0.00          O640  
ATOM   2115 O641 ROOT    1      31.382   4.227  16.586  0.00  0.00          O641  
ATOM   2116 O642 ROOT    1      29.323   2.556  17.062  0.00  0.00          O642  
ATOM   2117 O643 ROOT    1      28.977   0.372  15.523  0.00  0.00          O643  
ATOM   2118 O644 ROOT    1      27.257   2.438  15.336  0.00  0.00          O644  
ATOM   2119 O645 ROOT    1      26.466  -1.435  18.652  0.00  0.00          O645  
ATOM   2120 O646 ROOT    1      24.746   0.631  18.464  0.00  0.00          O646  
ATOM   2121 O647 ROOT    1      26.811   0.749  20.191  0.00  0.00          O647  
ATOM   2122 O649 ROOT    1      26.572  -3.484  14.460  0.00  0.00          O649  
ATOM   2123 O650 ROOT    1      31.142  -0.005  10.855  0.00  0.00          O650  
ATOM   2124 O651 ROOT    1      30.797  -2.190   9.316  0.00  0.00          O651  
ATOM   2125 O652 ROOT    1      29.077  -0.124   9.129  0.00  0.00          O652  
ATOM   2126 O653 ROOT    1      30.690  -0.141  13.508  0.00  0.00          O653  
ATOM   2127 O654 ROOT    1      28.631  -1.812  13.984  0.00  0.00          O654  
ATOM   2128 O655 ROOT    1      28.285  -3.997  12.445  0.00  0.00          O655  
ATOM   2129 O656 ROOT    1      26.566  -1.931  12.258  0.00  0.00          O656  
ATOM   2130 O657 ROOT    1      25.774  -5.804  15.573  0.00  0.00          O657  
ATOM   2131 O658 ROOT    1      24.054  -3.738  15.386  0.00  0.00          O658  
ATOM   2132 O659 ROOT    1      26.120  -3.619  17.112  0.00  0.00          O659  
ATOM   2133 O661 ROOT    1      30.703  -3.247  17.912  0.00  0.00          O661  
ATOM   2134 O662 ROOT    1      35.273   0.231  14.308  0.00  0.00          O662  
ATOM   2135 O663 ROOT    1      34.927  -1.953  12.769  0.00  0.00          O663  
ATOM   2136 O664 ROOT    1      33.208   0.113  12.581  0.00  0.00          O664  
ATOM   2137 O665 ROOT    1      34.821   0.096  16.960  0.00  0.00          O665  
ATOM   2138 O666 ROOT    1      32.762  -1.575  17.436  0.00  0.00          O666  
ATOM   2139 O667 ROOT    1      32.416  -3.760  15.897  0.00  0.00          O667  
ATOM   2140 O668 ROOT    1      30.696  -1.694  15.710  0.00  0.00          O668  
ATOM   2141 O669 ROOT    1      29.905  -5.567  19.026  0.00  0.00          O669  
ATOM   2142 O670 ROOT    1      28.185  -3.501  18.839  0.00  0.00          O670  
ATOM   2143 O671 ROOT    1      30.251  -3.382  20.565  0.00  0.00          O671  
ATOM   2144 O673 ROOT    1      30.011  -7.615  14.834  0.00  0.00          O673  
ATOM   2145 O674 ROOT    1      34.582  -4.137  11.229  0.00  0.00          O674  
ATOM   2146 O675 ROOT    1      34.236  -6.322   9.690  0.00  0.00          O675  
ATOM   2147 O676 ROOT    1      32.516  -4.256   9.503  0.00  0.00          O676  
ATOM   2148 O677 ROOT    1      34.130  -4.273  13.882  0.00  0.00          O677  
ATOM   2149 O678 ROOT    1      32.070  -5.944  14.358  0.00  0.00          O678  
ATOM   2150 O679 ROOT    1      31.725  -8.129  12.819  0.00  0.00          O679  
ATOM   2151 O680 ROOT    1      30.005  -6.063  12.632  0.00  0.00          O680  
ATOM   2152 O681 ROOT    1      29.213  -9.936  15.947  0.00  0.00          O681  
ATOM   2153 O682 ROOT    1      27.494  -7.870  15.760  0.00  0.00          O682  
ATOM   2154 O683 ROOT    1      29.559  -7.751  17.487  0.00  0.00          O683  
ATOM   2155 O685 ROOT    1      34.142  -7.378  18.286  0.00  0.00          O685  
ATOM   2156 O686 ROOT    1      38.712  -3.900  14.682  0.00  0.00          O686  
ATOM   2157 O687 ROOT    1      38.367  -6.085  13.143  0.00  0.00          O687  
ATOM   2158 O688 ROOT    1      36.647  -4.019  12.956  0.00  0.00          O688  
ATOM   2159 O689 ROOT    1      38.261  -4.036  17.335  0.00  0.00          O689  
ATOM   2160 O690 ROOT    1      36.201  -5.707  17.810  0.00  0.00          O690  
ATOM   2161 O691 ROOT    1      35.855  -7.892  16.271  0.00  0.00          O691  
ATOM   2162 O692 ROOT    1      34.136  -5.826  16.084  0.00  0.00          O692  
ATOM   2163 O693 ROOT    1      33.344  -9.699  19.400  0.00  0.00          O693  
ATOM   2164 O694 ROOT    1      31.624  -7.633  19.213  0.00  0.00          O694  
ATOM   2165 O695 ROOT    1      33.690  -7.514  20.939  0.00  0.00          O695  
ATOM   2166 O697 ROOT    1      33.450 -11.747  15.208  0.00  0.00          O697  
ATOM   2167 O698 ROOT    1      38.021  -8.269  11.604  0.00  0.00          O698  
ATOM   2168 O699 ROOT    1      37.675 -10.453  10.064  0.00  0.00          O699  
ATOM   2169 O700 ROOT    1      35.955  -8.387   9.877  0.00  0.00          O700  
ATOM   2170 O701 ROOT    1      37.569  -8.405  14.256  0.00  0.00          O701  
ATOM   2171 O702 ROOT    1      35.509 -10.076  14.732  0.00  0.00          O702  
ATOM   2172 O703 ROOT    1      35.164 -12.260  13.193  0.00  0.00          O703  
ATOM   2173 O704 ROOT    1      33.444 -10.194  13.006  0.00  0.00          O704  
ATOM   2174 O705 ROOT    1      32.653 -14.067  16.322  0.00  0.00          O705  
ATOM   2175 O706 ROOT    1      30.933 -12.001  16.135  0.00  0.00          O706  
ATOM   2176 O707 ROOT    1      32.998 -11.883  17.861  0.00  0.00          O707  
ATOM   2177 O709 ROOT    1      37.581 -11.510  18.661  0.00  0.00          O709  
ATOM   2178 O710 ROOT    1      42.152  -8.032  15.056  0.00  0.00          O710  
ATOM   2179 O711 ROOT    1      41.806 -10.216  13.517  0.00  0.00          O711  
ATOM   2180 O712 ROOT    1      40.086  -8.151  13.330  0.00  0.00          O712  
ATOM   2181 O713 ROOT    1      41.700  -8.168  17.709  0.00  0.00          O713  
ATOM   2182 O714 ROOT    1      39.640  -9.839  18.185  0.00  0.00          O714  
ATOM   2183 O715 ROOT    1      39.295 -12.023  16.646  0.00  0.00          O715  
ATOM   2184 O716 ROOT    1      37.575  -9.958  16.458  0.00  0.00          O716  
ATOM   2185 O717 ROOT    1      36.783 -13.830  19.774  0.00  0.00          O717  
ATOM   2186 O718 ROOT    1      35.064 -11.765  19.587  0.00  0.00          O718  
ATOM   2187 O719 ROOT    1      37.129 -11.646  21.313  0.00  0.00          O719  
ATOM   2188 O721 ROOT    1       7.319  30.045  18.371  0.00  0.00          O721  
ATOM   2189 O722 ROOT    1      11.890  33.523  14.766  0.00  0.00          O722  
ATOM   2190 O723 ROOT    1      11.544  31.338  13.227  0.00  0.00          O723  
ATOM   2191 O724 ROOT    1       9.825  33.404  13.040  0.00  0.00          O724  
ATOM   2192 O725 ROOT    1      11.438  33.387  17.419  0.00  0.00          O725  
ATOM   2193 O726 ROOT    1       9.379  31.716  17.895  0.00  0.00          O726  
ATOM   2194 O727 ROOT    1       9.033  29.531  16.356  0.00  0.00          O727  
ATOM   2195 O728 ROOT    1       7.313  31.597  16.169  0.00  0.00          O728  
ATOM   2196 O729 ROOT    1       6.522  27.724  19.485  0.00  0.00          O729  
ATOM   2197 O730 ROOT    1       4.802  29.790  19.297  0.00  0.00          O730  
ATOM   2198 O731 ROOT    1       6.867  29.909  21.024  0.00  0.00          O731  
ATOM   2199 O733 ROOT    1      11.450  30.281  21.823  0.00  0.00          O733  
ATOM   2200 O734 ROOT    1      16.021  33.760  18.219  0.00  0.00          O734  
ATOM   2201 O735 ROOT    1      15.675  31.575  16.680  0.00  0.00          O735  
ATOM   2202 O736 ROOT    1      13.955  33.641  16.493  0.00  0.00          O736  
ATOM   2203 O737 ROOT    1      15.569  33.624  20.872  0.00  0.00          O737  
ATOM   2204 O738 ROOT    1      13.510  31.953  21.348  0.00  0.00          O738  
ATOM   2205 O739 ROOT    1      13.164  29.768  19.808  0.00  0.00          O739  
ATOM   2206 O740 ROOT    1      11.444  31.834  19.621  0.00  0.00          O740  
ATOM   2207 O741 ROOT    1      10.653  27.961  22.937  0.00  0.00          O741  
ATOM   2208 O742 ROOT    1       8.933  30.027  22.750  0.00  0.00          O742  
ATOM   2209 O743 ROOT    1      10.998  30.146  24.476  0.00  0.00          O743  
ATOM   2210 O745 ROOT    1      10.759  25.913  18.745  0.00  0.00          O745  
ATOM   2211 O746 ROOT    1      15.329  29.391  15.141  0.00  0.00          O746  
ATOM   2212 O747 ROOT    1      14.983  27.206  13.601  0.00  0.00          O747  
ATOM   2213 O748 ROOT    1      13.264  29.272  13.414  0.00  0.00          O748  
ATOM   2214 O749 ROOT    1      14.877  29.255  17.793  0.00  0.00          O749  
ATOM   2215 O750 ROOT    1      12.818  27.584  18.269  0.00  0.00          O750  
ATOM   2216 O751 ROOT    1      12.472  25.400  16.730  0.00  0.00          O751  
ATOM   2217 O752 ROOT    1      10.753  27.465  16.543  0.00  0.00          O752  
ATOM   2218 O753 ROOT    1       9.961  23.593  19.859  0.00  0.00          O753  
ATOM   2219 O754 ROOT    1       8.241  25.658  19.672  0.00  0.00          O754  
ATOM   2220 O755 ROOT    1      10.307  25.777  21.398  0.00  0.00          O755  
ATOM   2221 O757 ROOT    1      14.889  26.150  22.198  0.00  0.00          O757  
ATOM   2222 O758 ROOT    1      19.460  29.628  18.593  0.00  0.00          O758  
ATOM   2223 O759 ROOT    1      19.114  27.443  17.054  0.00  0.00          O759  
ATOM   2224 O760 ROOT    1      17.395  29.509  16.867  0.00  0.00          O760  
ATOM   2225 O761 ROOT    1      19.008  29.492  21.246  0.00  0.00          O761  
ATOM   2226 O762 ROOT    1      16.949  27.821  21.722  0.00  0.00          O762  
ATOM   2227 O763 ROOT    1      16.603  25.636  20.183  0.00  0.00          O763  
ATOM   2228 O764 ROOT    1      14.883  27.702  19.996  0.00  0.00          O764  
ATOM   2229 O765 ROOT    1      14.092  23.829  23.311  0.00  0.00          O765  
ATOM   2230 O766 ROOT    1      12.372  25.895  23.124  0.00  0.00          O766  
ATOM   2231 O767 ROOT    1      14.438  26.014  24.850  0.00  0.00          O767  
ATOM   2232 O769 ROOT    1      14.198  21.781  19.119  0.00  0.00          O769  
ATOM   2233 O770 ROOT    1      18.768  25.259  15.515  0.00  0.00          O770  
ATOM   2234 O771 ROOT    1      18.423  23.075  13.976  0.00  0.00          O771  
ATOM   2235 O772 ROOT    1      16.703  25.141  13.789  0.00  0.00          O772  
ATOM   2236 O773 ROOT    1      18.317  25.123  18.168  0.00  0.00          O773  
ATOM   2237 O774 ROOT    1      16.257  23.452  18.644  0.00  0.00          O774  
ATOM   2238 O775 ROOT    1      15.911  21.268  17.104  0.00  0.00          O775  
ATOM   2239 O776 ROOT    1      14.192  23.334  16.917  0.00  0.00          O776  
ATOM   2240 O777 ROOT    1      13.400  19.461  20.233  0.00  0.00          O777  
ATOM   2241 O778 ROOT    1      11.681  21.527  20.046  0.00  0.00          O778  
ATOM   2242 O779 ROOT    1      13.746  21.645  21.772  0.00  0.00          O779  
ATOM   2243 O781 ROOT    1      18.329  22.018  22.572  0.00  0.00          O781  
ATOM   2244 O782 ROOT    1      22.899  25.496  18.967  0.00  0.00          O782  
ATOM   2245 O783 ROOT    1      22.553  23.312  17.428  0.00  0.00          O783  
ATOM   2246 O784 ROOT    1      20.834  25.378  17.241  0.00  0.00          O784  
ATOM   2247 O785 ROOT    1      22.447  25.360  21.620  0.00  0.00          O785  
ATOM   2248 O786 ROOT    1      20.388  23.689  22.096  0.00  0.00          O786  
ATOM   2249 O787 ROOT    1      20.042  21.505  20.557  0.00  0.00          O787  
ATOM   2250 O788 ROOT    1      18.323  23.571  20.370  0.00  0.00          O788  
ATOM   2251 O789 ROOT    1      17.531  19.698  23.686  0.00  0.00          O789  
ATOM   2252 O790 ROOT    1      15.811  21.764  23.498  0.00  0.00          O790  
ATOM   2253 O791 ROOT    1      17.877  21.882  25.225  0.00  0.00          O791  
ATOM   2254 O793 ROOT    1      17.637  17.649  19.494  0.00  0.00          O793  
ATOM   2255 O794 ROOT    1      22.208  21.127  15.889  0.00  0.00          O794  
ATOM   2256 O795 ROOT    1      21.862  18.943  14.350  0.00  0.00          O795  
ATOM   2257 O796 ROOT    1      20.142  21.009  14.163  0.00  0.00          O796  
ATOM   2258 O797 ROOT    1      21.756  20.991  18.542  0.00  0.00          O797  
ATOM   2259 O798 ROOT    1      19.696  19.320  19.018  0.00  0.00          O798  
ATOM   2260 O799 ROOT    1      19.351  17.136  17.479  0.00  0.00          O799  
ATOM   2261 O800 ROOT    1      17.631  19.202  17.291  0.00  0.00          O800  
ATOM   2262 O801 ROOT    1      16.839  15.329  20.607  0.00  0.00          O801  
ATOM   2263 O802 ROOT    1      15.120  17.395  20.420  0.00  0.00          O802  
ATOM   2264 O803 ROOT    1      17.185  17.513  22.146  0.00  0.00          O803  
ATOM   2265 O805 ROOT    1      21.768  17.886  22.946  0.00  0.00          O805  
ATOM   2266 O806 ROOT    1      26.339  21.364  19.342  0.00  0.00          O806  
ATOM   2267 O807 ROOT    1      25.993  19.180  17.802  0.00  0.00          O807  
ATOM   2268 O808 ROOT    1      24.273  21.246  17.615  0.00  0.00          O808  
ATOM   2269 O809 ROOT    1      25.887  21.228  21.994  0.00  0.00          O809  
ATOM   2270 O810 ROOT    1      23.827  19.557  22.470  0.00  0.00          O810  
ATOM   2271 O811 ROOT    1      23.482  17.373  20.931  0.00  0.00          O811  
ATOM   2272 O812 ROOT    1      21.762  19.439  20.744  0.00  0.00          O812  
ATOM   2273 O813 ROOT    1      20.970  15.566  24.060  0.00  0.00          O813  
ATOM   2274 O814 ROOT    1      19.251  17.632  23.873  0.00  0.00          O814  
ATOM   2275 O815 ROOT    1      21.316  17.750  25.599  0.00  0.00          O815  
ATOM   2276 O817 ROOT    1      21.076  13.517  19.868  0.00  0.00          O817  
ATOM   2277 O818 ROOT    1      25.647  16.996  16.263  0.00  0.00          O818  
ATOM   2278 O819 ROOT    1      25.301  14.811  14.724  0.00  0.00          O819  
ATOM   2279 O820 ROOT    1      23.581  16.877  14.537  0.00  0.00          O820  
ATOM   2280 O821 ROOT    1      25.195  16.860  18.916  0.00  0.00          O821  
ATOM   2281 O822 ROOT    1      23.136  15.189  19.392  0.00  0.00          O822  
ATOM   2282 O823 ROOT    1      22.790  13.004  17.853  0.00  0.00          O823  
ATOM   2283 O824 ROOT    1      21.070  15.070  17.666  0.00  0.00          O824  
ATOM   2284 O825 ROOT    1      20.279  11.197  20.981  0.00  0.00          O825  
ATOM   2285 O826 ROOT    1      18.559  13.263  20.794  0.00  0.00          O826  
ATOM   2286 O827 ROOT    1      20.624  13.382  22.521  0.00  0.00          O827  
ATOM   2287 O829 ROOT    1      25.207  13.754  23.320  0.00  0.00          O829  
ATOM   2288 O830 ROOT    1      29.778  17.232  19.716  0.00  0.00          O830  
ATOM   2289 O831 ROOT    1      29.432  15.048  18.177  0.00  0.00          O831  
ATOM   2290 O832 ROOT    1      27.712  17.114  17.990  0.00  0.00          O832  
ATOM   2291 O833 ROOT    1      29.326  17.097  22.369  0.00  0.00          O833  
ATOM   2292 O834 ROOT    1      27.267  15.425  22.844  0.00  0.00          O834  
ATOM   2293 O835 ROOT    1      26.921  13.241  21.305  0.00  0.00          O835  
ATOM   2294 O836 ROOT    1      25.201  15.307  21.118  0.00  0.00          O836  
ATOM   2295 O837 ROOT    1      24.410  11.434  24.434  0.00  0.00          O837  
ATOM   2296 O838 ROOT    1      22.690  13.500  24.247  0.00  0.00          O838  
ATOM   2297 O839 ROOT    1      24.755  13.618  25.973  0.00  0.00          O839  
ATOM   2298 O841 ROOT    1      24.516   9.386  20.242  0.00  0.00          O841  
ATOM   2299 O842 ROOT    1      29.086  12.864  16.638  0.00  0.00          O842  
ATOM   2300 O843 ROOT    1      28.740  10.679  15.098  0.00  0.00          O843  
ATOM   2301 O844 ROOT    1      27.021  12.745  14.911  0.00  0.00          O844  
ATOM   2302 O845 ROOT    1      28.634  12.728  19.290  0.00  0.00          O845  
ATOM   2303 O846 ROOT    1      26.575  11.057  19.766  0.00  0.00          O846  
ATOM   2304 O847 ROOT    1      26.229   8.872  18.227  0.00  0.00          O847  
ATOM   2305 O848 ROOT    1      24.510  10.938  18.040  0.00  0.00          O848  
ATOM   2306 O849 ROOT    1      23.718   7.065  21.356  0.00  0.00          O849  
ATOM   2307 O850 ROOT    1      21.998   9.131  21.169  0.00  0.00          O850  
ATOM   2308 O851 ROOT    1      24.064   9.250  22.895  0.00  0.00          O851  
ATOM   2309 O853 ROOT    1      28.646   9.623  23.695  0.00  0.00          O853  
ATOM   2310 O854 ROOT    1      33.217  13.101  20.090  0.00  0.00          O854  
ATOM   2311 O855 ROOT    1      32.871  10.916  18.551  0.00  0.00          O855  
ATOM   2312 O856 ROOT    1      31.152  12.982  18.364  0.00  0.00          O856  
ATOM   2313 O857 ROOT    1      32.765  12.965  22.743  0.00  0.00          O857  
ATOM   2314 O858 ROOT    1      30.706  11.294  23.219  0.00  0.00          O858  
ATOM   2315 O859 ROOT    1      30.360   9.109  21.680  0.00  0.00          O859  
ATOM   2316 O860 ROOT    1      28.640  11.175  21.492  0.00  0.00          O860  
ATOM   2317 O861 ROOT    1      27.849   7.302  24.808  0.00  0.00          O861  
ATOM   2318 O862 ROOT    1      26.129   9.368  24.621  0.00  0.00          O862  
ATOM   2319 O863 ROOT    1      28.195   9.487  26.347  0.00  0.00          O863  
ATOM   2320 O865 ROOT    1      27.955   5.254  20.616  0.00  0.00          O865  
ATOM   2321 O866 ROOT    1      32.525   8.732  17.012  0.00  0.00          O866  
ATOM   2322 O867 ROOT    1      32.180   6.548  15.473  0.00  0.00          O867  
ATOM   2323 O868 ROOT    1      30.460   8.613  15.285  0.00  0.00          O868  
ATOM   2324 O869 ROOT    1      32.074   8.596  19.664  0.00  0.00          O869  
ATOM   2325 O870 ROOT    1      30.014   6.925  20.140  0.00  0.00          O870  
ATOM   2326 O871 ROOT    1      29.668   4.741  18.601  0.00  0.00          O871  
ATOM   2327 O872 ROOT    1      27.949   6.807  18.414  0.00  0.00          O872  
ATOM   2328 O873 ROOT    1      27.157   2.934  21.730  0.00  0.00          O873  
ATOM   2329 O874 ROOT    1      25.438   5.000  21.543  0.00  0.00          O874  
ATOM   2330 O875 ROOT    1      27.503   5.118  23.269  0.00  0.00          O875  
ATOM   2331 O877 ROOT    1      32.086   5.491  24.069  0.00  0.00          O877  
ATOM   2332 O878 ROOT    1      36.656   8.969  20.464  0.00  0.00          O878  
ATOM   2333 O879 ROOT    1      36.311   6.785  18.925  0.00  0.00          O879  
ATOM   2334 O880 ROOT    1      34.591   8.850  18.738  0.00  0.00          O880  
ATOM   2335 O881 ROOT    1      36.204   8.833  23.117  0.00  0.00          O881  
ATOM   2336 O882 ROOT    1      34.145   7.162  23.593  0.00  0.00          O882  
ATOM   2337 O883 ROOT    1      33.799   4.978  22.054  0.00  0.00          O883  
ATOM   2338 O884 ROOT    1      32.080   7.044  21.867  0.00  0.00          O884  
ATOM   2339 O885 ROOT    1      31.288   3.171  25.182  0.00  0.00          O885  
ATOM   2340 O886 ROOT    1      29.568   5.237  24.995  0.00  0.00          O886  
ATOM   2341 O887 ROOT    1      31.634   5.355  26.722  0.00  0.00          O887  
ATOM   2342 O889 ROOT    1      31.394   1.122  20.991  0.00  0.00          O889  
ATOM   2343 O890 ROOT    1      35.965   4.600  17.386  0.00  0.00          O890  
ATOM   2344 O891 ROOT    1      35.619   2.416  15.847  0.00  0.00          O891  
ATOM   2345 O892 ROOT    1      33.899   4.482  15.660  0.00  0.00          O892  
ATOM   2346 O893 ROOT    1      35.513   4.464  20.039  0.00  0.00          O893  
ATOM   2347 O894 ROOT    1      33.453   2.793  20.515  0.00  0.00          O894  
ATOM   2348 O895 ROOT    1      33.108   0.609  18.975  0.00  0.00          O895  
ATOM   2349 O896 ROOT    1      31.388   2.675  18.788  0.00  0.00          O896  
ATOM   2350 O897 ROOT    1      30.596  -1.198  22.104  0.00  0.00          O897  
ATOM   2351 O898 ROOT    1      28.877   0.868  21.917  0.00  0.00          O898  
ATOM   2352 O899 ROOT    1      30.942   0.986  23.643  0.00  0.00          O899  
ATOM   2353 O901 ROOT    1      35.525   1.359  24.443  0.00  0.00          O901  
ATOM   2354 O902 ROOT    1      40.096   4.837  20.838  0.00  0.00          O902  
ATOM   2355 O903 ROOT    1      39.750   2.653  19.299  0.00  0.00          O903  
ATOM   2356 O904 ROOT    1      38.030   4.719  19.112  0.00  0.00          O904  
ATOM   2357 O905 ROOT    1      39.644   4.701  23.491  0.00  0.00          O905  
ATOM   2358 O906 ROOT    1      37.584   3.030  23.967  0.00  0.00          O906  
ATOM   2359 O907 ROOT    1      37.239   0.846  22.428  0.00  0.00          O907  
ATOM   2360 O908 ROOT    1      35.519   2.912  22.241  0.00  0.00          O908  
ATOM   2361 O909 ROOT    1      34.727  -0.961  25.557  0.00  0.00          O909  
ATOM   2362 O910 ROOT    1      33.008   1.105  25.370  0.00  0.00          O910  
ATOM   2363 O911 ROOT    1      35.073   1.223  27.096  0.00  0.00          O911  
ATOM   2364 O913 ROOT    1      34.833  -3.010  21.365  0.00  0.00          O913  
ATOM   2365 O914 ROOT    1      39.404   0.468  17.760  0.00  0.00          O914  
ATOM   2366 O915 ROOT    1      39.058  -1.716  16.221  0.00  0.00          O915  
ATOM   2367 O916 ROOT    1      37.338   0.350  16.034  0.00  0.00          O916  
ATOM   2368 O917 ROOT    1      38.952   0.332  20.413  0.00  0.00          O917  
ATOM   2369 O918 ROOT    1      36.893  -1.339  20.889  0.00  0.00          O918  
ATOM   2370 O919 ROOT    1      36.547  -3.523  19.350  0.00  0.00          O919  
ATOM   2371 O920 ROOT    1      34.827  -1.457  19.163  0.00  0.00          O920  
ATOM   2372 O921 ROOT    1      34.036  -5.330  22.478  0.00  0.00          O921  
ATOM   2373 O922 ROOT    1      32.316  -3.264  22.291  0.00  0.00          O922  
ATOM   2374 O923 ROOT    1      34.381  -3.146  24.017  0.00  0.00          O923  
ATOM   2375 O925 ROOT    1      38.964  -2.773  24.817  0.00  0.00          O925  
ATOM   2376 O926 ROOT    1      43.535   0.705  21.213  0.00  0.00          O926  
ATOM   2377 O927 ROOT    1      43.189  -1.479  19.674  0.00  0.00          O927  
ATOM   2378 O928 ROOT    1      41.469   0.587  19.486  0.00  0.00          O928  
ATOM   2379 O929 ROOT    1      43.083   0.569  23.865  0.00  0.00          O929  
ATOM   2380 O930 ROOT    1      41.024  -1.102  24.341  0.00  0.00          O930  
ATOM   2381 O931 ROOT    1      40.678  -3.286  22.802  0.00  0.00          O931  
ATOM   2382 O932 ROOT    1      38.958  -1.220  22.615  0.00  0.00          O932  
ATOM   2383 O933 ROOT    1      38.167  -5.093  25.931  0.00  0.00          O933  
ATOM   2384 O934 ROOT    1      36.447  -3.027  25.744  0.00  0.00          O934  
ATOM   2385 O935 ROOT    1      38.512  -2.909  27.470  0.00  0.00          O935  
ATOM   2386 O937 ROOT    1      38.273  -7.141  21.739  0.00  0.00          O937  
ATOM   2387 O938 ROOT    1      42.843  -3.663  18.134  0.00  0.00          O938  
ATOM   2388 O939 ROOT    1      42.497  -5.848  16.595  0.00  0.00          O939  
ATOM   2389 O940 ROOT    1      40.778  -3.782  16.408  0.00  0.00          O940  
ATOM   2390 O941 ROOT    1      42.391  -3.799  20.787  0.00  0.00          O941  
ATOM   2391 O942 ROOT    1      40.332  -5.470  21.263  0.00  0.00          O942  
ATOM   2392 O943 ROOT    1      39.986  -7.655  19.724  0.00  0.00          O943  
ATOM   2393 O944 ROOT    1      38.267  -5.589  19.537  0.00  0.00          O944  
ATOM   2394 O945 ROOT    1      37.475  -9.462  22.852  0.00  0.00          O945  
ATOM   2395 O946 ROOT    1      35.755  -7.396  22.665  0.00  0.00          O946  
ATOM   2396 O947 ROOT    1      37.821  -7.277  24.392  0.00  0.00          O947  
ATOM   2397 O949 ROOT    1      42.403  -6.904  25.191  0.00  0.00          O949  
ATOM   2398 O950 ROOT    1      46.974  -3.426  21.587  0.00  0.00          O950  
ATOM   2399 O951 ROOT    1      46.628  -5.611  20.048  0.00  0.00          O951  
ATOM   2400 O952 ROOT    1      44.909  -3.545  19.861  0.00  0.00          O952  
ATOM   2401 O953 ROOT    1      46.522  -3.562  24.240  0.00  0.00          O953  
ATOM   2402 O954 ROOT    1      44.463  -5.233  24.716  0.00  0.00          O954  
ATOM   2403 O955 ROOT    1      44.117  -7.418  23.176  0.00  0.00          O955  
ATOM   2404 O956 ROOT    1      42.397  -5.352  22.989  0.00  0.00          O956  
ATOM   2405 O957 ROOT    1      41.606  -9.225  26.305  0.00  0.00          O957  
ATOM   2406 O958 ROOT    1      39.886  -7.159  26.118  0.00  0.00          O958  
ATOM   2407 O959 ROOT    1      41.952  -7.040  27.844  0.00  0.00          O959  
ATOM   2408 O961 ROOT    1      12.142  34.650  24.902  0.00  0.00          O961  
ATOM   2409 O962 ROOT    1      16.712  38.128  21.297  0.00  0.00          O962  
ATOM   2410 O963 ROOT    1      16.367  35.944  19.758  0.00  0.00          O963  
ATOM   2411 O964 ROOT    1      14.647  38.010  19.571  0.00  0.00          O964  
ATOM   2412 O965 ROOT    1      16.260  37.992  23.950  0.00  0.00          O965  
ATOM   2413 O966 ROOT    1      14.201  36.321  24.426  0.00  0.00          O966  
ATOM   2414 O967 ROOT    1      13.855  34.137  22.887  0.00  0.00          O967  
ATOM   2415 O968 ROOT    1      12.136  36.203  22.700  0.00  0.00          O968  
ATOM   2416 O969 ROOT    1      11.344  32.330  26.015  0.00  0.00          O969  
ATOM   2417 O970 ROOT    1       9.624  34.396  25.828  0.00  0.00          O970  
ATOM   2418 O971 ROOT    1      11.690  34.514  27.555  0.00  0.00          O971  
ATOM   2419 O973 ROOT    1      16.273  34.887  28.354  0.00  0.00          O973  
ATOM   2420 O974 ROOT    1      20.843  38.365  24.750  0.00  0.00          O974  
ATOM   2421 O975 ROOT    1      20.497  36.181  23.211  0.00  0.00          O975  
ATOM   2422 O976 ROOT    1      18.778  38.247  23.023  0.00  0.00          O976  
ATOM   2423 O977 ROOT    1      20.391  38.229  27.403  0.00  0.00          O977  
ATOM   2424 O978 ROOT    1      18.332  36.558  27.878  0.00  0.00          O978  
ATOM   2425 O979 ROOT    1      17.986  34.374  26.339  0.00  0.00          O979  
ATOM   2426 O980 ROOT    1      16.266  36.440  26.152  0.00  0.00          O980  
ATOM   2427 O981 ROOT    1      15.475  32.567  29.468  0.00  0.00          O981  
ATOM   2428 O982 ROOT    1      13.755  34.633  29.281  0.00  0.00          O982  
ATOM   2429 O983 ROOT    1      15.821  34.751  31.007  0.00  0.00          O983  
ATOM   2430 O985 ROOT    1      15.581  30.518  25.276  0.00  0.00          O985  
ATOM   2431 O986 ROOT    1      20.152  33.996  21.671  0.00  0.00          O986  
ATOM   2432 O987 ROOT    1      19.806  31.812  20.132  0.00  0.00          O987  
ATOM   2433 O988 ROOT    1      18.086  33.878  19.945  0.00  0.00          O988  
ATOM   2434 O989 ROOT    1      19.700  33.861  24.324  0.00  0.00          O989  
ATOM   2435 O990 ROOT    1      17.640  32.190  24.800  0.00  0.00          O990  
ATOM   2436 O991 ROOT    1      17.295  30.005  23.261  0.00  0.00          O991  
ATOM   2437 O992 ROOT    1      15.575  32.071  23.074  0.00  0.00          O992  
ATOM   2438 O993 ROOT    1      14.783  28.198  26.390  0.00  0.00          O993  
ATOM   2439 O994 ROOT    1      13.064  30.264  26.202  0.00  0.00          O994  
ATOM   2440 O995 ROOT    1      15.129  30.383  27.929  0.00  0.00          O995  
ATOM   2441 O997 ROOT    1      19.712  30.755  28.729  0.00  0.00          O997  
ATOM   2442 O998 ROOT    1      24.282  34.233  25.124  0.00  0.00          O998  
ATOM   2443 O999 ROOT    1      23.937  32.049  23.585  0.00  0.00          O999  
CONECT    1    3    7    9
CONECT    2    5   23   24
CONECT    3    1    4   25
CONECT    4    3    5   22
CONECT    5    2    4    6
CONECT    6    5    7    8
CONECT    7    1    6   26
CONECT    8    6   18   19   20
CONECT    9    1   10   11   15
CONECT   10    9   16   17   21
CONECT   11    9   12   13   14
CONECT   12   11
CONECT   13   11
CONECT   14   11
CONECT   15    9
CONECT   16   10
CONECT   17   10
CONECT   18    8
CONECT   19    8
CONECT   20    8
CONECT   21   10
CONECT   22    4
CONECT   23    2
CONECT   24    2
CONECT   25    3
CONECT   26    7
CONECT   27   28   29
CONECT   28   27
CONECT   29   27
CONECT   30   31   32
CONECT   31   30
CONECT   32   30
CONECT   33   34   35
CONECT   34   33
CONECT   35   33
CONECT   36   37   38
CONECT   37   36
CONECT   38   36
CONECT   39   40   41
CONECT   40   39
CONECT   41   39
CONECT   42   43   44
CONECT   43   42
CONECT   44   42
CONECT   45   46   47
CONECT   46   45
CONECT   47   45
CONECT   48   49   50
CONECT   49   48
CONECT   50   48
CONECT   51   52   53
CONECT   52   51
CONECT   53   51
CONECT   54   55   56
CONECT   55   54
CONECT   56   54
CONECT   57   58   59
CONECT   58   57
CONECT   59   57
CONECT   60   61   62
CONECT   61   60
CONECT   62   60
CONECT   63   64   65
CONECT   64   63
CONECT   65   63
CONECT   66   67   68
CONECT   67   66
CONECT   68   66
CONECT   69   70   71
CONECT   70   69
CONECT   71   69
CONECT   72   83  148  149
CONECT   73   89  154  155
CONECT   74   95  160  161
CONECT   75  101  166  167
CONECT   76  107  172  173
CONECT   77  113  178  179
CONECT   78  119  184  185
CONECT   79  125  190  191
CONECT   80   81   85  140
CONECT   81   80   82  150
CONECT   82   81   83  151
CONECT   83   72   82   84
CONECT   84   83   85  152
CONECT   85   80   84  153
CONECT   86   87   91  141
CONECT   87   86   88  156
CONECT   88   87   89  157
CONECT   89   73   88   90
CONECT   90   89   91  158
CONECT   91   86   90  159
CONECT   92   93   97  142
CONECT   93   92   94  162
CONECT   94   93   95  163
CONECT   95   74   94   96
CONECT   96   95   97  164
CONECT   97   92   96  165
CONECT   98   99  103  143
CONECT   99   98  100  168
CONECT  100   99  101  169
CONECT  101   75  100  102
CONECT  102  101  103  170
CONECT  103   98  102  171
CONECT  104  105  109  144
CONECT  105  104  106  174
CONECT  106  105  107  175
CONECT  107   76  106  108
CONECT  108  107  109  176
CONECT  109  104  108  177
CONECT  110  111  115  145
CONECT  111  110  112  180
CONECT  112  111  113  181
CONECT  113   77  112  114
CONECT  114  113  115  182
CONECT  115  110  114  183
CONECT  116  117  121  146
CONECT  117  116  118  186
CONECT  118  117  119  187
CONECT  119   78  118  120
CONECT  120  119  121  188
CONECT  121  116  120  189
CONECT  122  123  127  147
CONECT  123  122  124  192
CONECT  124  123  125  193
CONECT  125   79  124  126
CONECT  126  125  127  194
CONECT  127  122  126  195
CONECT  128  140  141
CONECT  129  142  143
CONECT  130  144  145
CONECT  131  146  147
CONECT  132  140  142
CONECT  133  141  143
CONECT  134  142  144
CONECT  135  143  145
CONECT  136  144  146
CONECT  137  145  147
CONECT  138  140  146
CONECT  139  141  147
CONECT  140   80  128  132  138
CONECT  141   86  128  133  139
CONECT  142   92  129  132  134
CONECT  143   98  129  133  135
CONECT  144  104  130  134  136
CONECT  145  110  130  135  137
CONECT  146  116  131  136  138
CONECT  147  122  131  137  139
CONECT  148   72
CONECT  149   72
CONECT  150   81
CONECT  151   82
CONECT  152   84
CONECT  153   85
CONECT  154   73
CONECT  155   73
CONECT  156   87
CONECT  157   88
CONECT  158   90
CONECT  159   91
CONECT  160   74
CONECT  161   74
CONECT  162   93
CONECT  163   94
CONECT  164   96
CONECT  165   97
CONECT  166   75
CONECT  167   75
CONECT  168   99
CONECT  169  100
CONECT  170  102
CONECT  171  103
CONECT  172   76
CONECT  173   76
CONECT  174  105
CONECT  175  106
CONECT  176  108
CONECT  177  109
CONECT  178   77
CONECT  179   77
CONECT  180  111
CONECT  181  112
CONECT  182  114
CONECT  183  115
CONECT  184   78
CONECT  185   78
CONECT  186  117
CONECT  187  118
CONECT  188  120
CONECT  189  121
CONECT  190   79
CONECT  191   79
CONECT  192  123
CONECT  193  124
CONECT  194  126
CONECT  195  127
CONECT  196  197  200
CONECT  197  196
CONECT  198  199  200
CONECT  199  198
CONECT  200  196  198  211  212
CONECT  201  202  215
CONECT  202  201
CONECT  203  204  216
CONECT  204  203
CONECT  205  206  217
CONECT  206  205
CONECT  207  208  218
CONECT  208  207
CONECT  209  210  215
CONECT  210  209
CONECT  211  200  216
CONECT  212  200  217
CONECT  213  214  218
CONECT  214  213
CONECT  215  201  209  223  247
CONECT  216  203  211  225  248
CONECT  217  205  212  230  249
CONECT  218  207  213  231  250
CONECT  219  220  237
CONECT  220  219
CONECT  221  222  238
CONECT  222  221
CONECT  223  215  224
CONECT  224  223
CONECT  225  216  237
CONECT  226  227  238
CONECT  227  226
CONECT  228  229  239
CONECT  229  228
CONECT  230  217  240
CONECT  231  218  232
CONECT  232  231
CONECT  233  234  239
CONECT  234  233
CONECT  235  236  240
CONECT  236  235
CONECT  237  219  225  259  264
CONECT  238  221  226  260  265
CONECT  239  228  233  266  270
CONECT  240  230  235  267  272
CONECT  241  242  275
CONECT  242  241
CONECT  243  244  276
CONECT  244  243
CONECT  245  246  277
CONECT  246  245
CONECT  247  215  278
CONECT  248  216  279
CONECT  249  217  280
CONECT  250  218  281
CONECT  251  252  282
CONECT  252  251
CONECT  253  254  283
CONECT  254  253
CONECT  255  256  284
CONECT  256  255
CONECT  257  258  275
CONECT  258  257
CONECT  259  237  276
CONECT  260  238  261
CONECT  261  260
CONECT  262  263  277
CONECT  263  262
CONECT  264  237  278
CONECT  265  238  279
CONECT  266  239  280
CONECT  267  240  281
CONECT  268  269  282
CONECT  269  268
CONECT  270  239  271
CONECT  271  270
CONECT  272  240  283
CONECT  273  274  284
CONECT  274  273
CONECT  275  241  257  285  302
CONECT  276  243  259  287  303
CONECT  277  245  262  289  304
CONECT  278  247  264  290  306
CONECT  279  248  265  291  307
CONECT  280  249  266  292  312
CONECT  281  250  267  293  313
CONECT  282  251  268  294  314
CONECT  283  253  272  295  316
CONECT  284  255  273  297  317
CONECT  285  275  286
CONECT  286  285
CONECT  287  276  288
CONECT  288  287
CONECT  289  277  299
CONECT  290  278  300
CONECT  291  279  301
CONECT  292  280  299
CONECT  293  281  300
CONECT  294  282  301
CONECT  295  283  296
CONECT  296  295
CONECT  297  284  298
CONECT  298  297
CONECT  299  289  292  322  325
CONECT  300  290  293  323  326
CONECT  301  291  294  324  327
CONECT  302  275  332
CONECT  303  276  333
CONECT  304  277  305
CONECT  305  304
CONECT  306  278  334
CONECT  307  279  335
CONECT  308  309  336
CONECT  309  308
CONECT  310  311  337
CONECT  311  310
CONECT  312  280  338
CONECT  313  281  339
CONECT  314  282  315
CONECT  315  314
CONECT  316  283  340
CONECT  317  284  341
CONECT  318  319  332
CONECT  319  318
CONECT  320  321  333
CONECT  321  320
CONECT  322  299  334
CONECT  323  300  335
CONECT  324  301  336
CONECT  325  299  337
CONECT  326  300  338
CONECT  327  301  339
CONECT  328  329  340
CONECT  329  328
CONECT  330  331  341
CONECT  331  330
CONECT  332  302  318  344  362
CONECT  333  303  320  345  363
CONECT  334  306  322  346  366
CONECT  335  307  323  347  367
CONECT  336  308  324  348  368
CONECT  337  310  325  349  370
CONECT  338  312  326  350  372
CONECT  339  313  327  351  373
CONECT  340  316  328  352  376
CONECT  341  317  330  353  377
CONECT  342  343  356
CONECT  343  342
CONECT  344  332  357
CONECT  345  333  358
CONECT  346  334  356
CONECT  347  335  357
CONECT  348  336  358
CONECT  349  337  359
CONECT  350  338  360
CONECT  351  339  361
CONECT  352  340  359
CONECT  353  341  360
CONECT  354  355  361
CONECT  355  354
CONECT  356  342  346  378  382
CONECT  357  344  347  379  383
CONECT  358  345  348  380  384
CONECT  359  349  352  385  388
CONECT  360  350  353  386  390
CONECT  361  351  354  387  391
CONECT  362  332  392
CONECT  363  333  393
CONECT  364  365  394
CONECT  365  364
CONECT  366  334  395
CONECT  367  335  396
CONECT  368  336  369
CONECT  369  368
CONECT  370  337  371
CONECT  371  370
CONECT  372  338  397
CONECT  373  339  398
CONECT  374  375  399
CONECT  375  374
CONECT  376  340  400
CONECT  377  341  401
CONECT  378  356  392
CONECT  379  357  393
CONECT  380  358  381
CONECT  381  380
CONECT  382  356  394
CONECT  383  357  395
CONECT  384  358  396
CONECT  385  359  397
CONECT  386  360  398
CONECT  387  361  399
CONECT  388  359  389
CONECT  389  388
CONECT  390  360  400
CONECT  391  361  401
CONECT  392  362  378  404  423
CONECT  393  363  379  406  424
CONECT  394  364  382  407  425
CONECT  395  366  383  408  427
CONECT  396  367  384  409  428
CONECT  397  372  385  410  433
CONECT  398  373  386  411  434
CONECT  399  374  387  412  435
CONECT  400  376  390  413  437
CONECT  401  377  391  414  438
CONECT  402  403  418
CONECT  403  402
CONECT  404  392  405
CONECT  405  404
CONECT  406  393  418
CONECT  407  394  419
CONECT  408  395  420
CONECT  409  396  421
CONECT  410  397  419
CONECT  411  398  420
CONECT  412  399  421
CONECT  413  400  422
CONECT  414  401  415
CONECT  415  414
CONECT  416  417  422
CONECT  417  416
CONECT  418  402  406  439  441
CONECT  419  407  410  444  447
CONECT  420  408  411  445  448
CONECT  421  409  412  446  449
CONECT  422  413  416  452  453
CONECT  423  392  455
CONECT  424  393  456
CONECT  425  394  426
CONECT  426  425
CONECT  427  395  457
CONECT  428  396  458
CONECT  429  430  459
CONECT  430  429
CONECT  431  432  460
CONECT  432  431
CONECT  433  397  461
CONECT  434  398  462
CONECT  435  399  436
CONECT  436  435
CONECT  437  400  463
CONECT  438  401  464
CONECT  439  418  440
CONECT  440  439
CONECT  441  418  455
CONECT  442  443  456
CONECT  443  442
CONECT  444  419  457
CONECT  445  420  458
CONECT  446  421  459
CONECT  447  419  460
CONECT  448  420  461
CONECT  449  421  462
CONECT  450  451  463
CONECT  451  450
CONECT  452  422  464
CONECT  453  422  454
CONECT  454  453
CONECT  455  423  441  467  485
CONECT  456  424  442  468  487
CONECT  457  427  444  469  489
CONECT  458  428  445  470  490
CONECT  459  429  446  471  491
CONECT  460  431  447  472  493
CONECT  461  433  448  473  495
CONECT  462  434  449  474  496
CONECT  463  437  450  475  497
CONECT  464  438  452  476  499
CONECT  465  466  479
CONECT  466  465
CONECT  467  455  480
CONECT  468  456  481
CONECT  469  457  479
CONECT  470  458  480
CONECT  471  459  481
CONECT  472  460  482
CONECT  473  461  483
CONECT  474  462  484
CONECT  475  463  482
CONECT  476  464  483
CONECT  477  478  484
CONECT  478  477
CONECT  479  465  469  501  507
CONECT  480  467  470  503  509
CONECT  481  468  471  505  510
CONECT  482  472  475  511  515
CONECT  483  473  476  512  517
CONECT  484  474  477  513  519
CONECT  485  455  486
CONECT  486  485
CONECT  487  456  488
CONECT  488  487
CONECT  489  457  521
CONECT  490  458  522
CONECT  491  459  492
CONECT  492  491
CONECT  493  460  494
CONECT  494  493
CONECT  495  461  523
CONECT  496  462  524
CONECT  497  463  498
CONECT  498  497
CONECT  499  464  500
CONECT  500  499
CONECT  501  479  502
CONECT  502  501
CONECT  503  480  504
CONECT  504  503
CONECT  505  481  506
CONECT  506  505
CONECT  507  479  508
CONECT  508  507
CONECT  509  480  521
CONECT  510  481  522
CONECT  511  482  523
CONECT  512  483  524
CONECT  513  484  514
CONECT  514  513
CONECT  515  482  516
CONECT  516  515
CONECT  517  483  518
CONECT  518  517
CONECT  519  484  520
CONECT  520  519
CONECT  521  489  509  525  532
CONECT  522  490  510  526  534
CONECT  523  495  511  528  536
CONECT  524  496  512  530  538
CONECT  525  521  531
CONECT  526  522  527
CONECT  527  526
CONECT  528  523  529
CONECT  529  528
CONECT  530  524  531
CONECT  531  525  530  540  542
CONECT  532  521  533
CONECT  533  532
CONECT  534  522  535
CONECT  535  534
CONECT  536  523  537
CONECT  537  536
CONECT  538  524  539
CONECT  539  538
CONECT  540  531  541
CONECT  541  540
CONECT  542  531  543
CONECT  543  542
END`, "pdb", {{assignBonds:true,}});
        v.setStyle({{}},{{sphere:{{scale:0.25, colorscheme:'Jmol'}}, stick:{{radius:0.15, colorscheme:'Jmol'}}}});
        v.setBackgroundColor(0xffffff, 0.0);
        v.zoomTo();
        v.render(/* no callback */ );
        </script>
         <style>
            .dataframe {{
                border-collapse: collapse;
                font-size: 0.9em;
                font-family: "Source Sans Pro",sans-serif;
                min-width: 100%;
                border-radius: 10px;
                overflow:hidden;
            }}
            .dataframe thead th {{
                background-color: #d4edf7;
                font-family: "Source Sans Pro",sans-serif;
                color: #000000;
                text-align: center;
            }}
            .dataframe thead th:nth-child(1) {{
                background-color: #d4edf7;
                font-family: "Source Sans Pro",sans-serif;
                color: #000000;
                text-align: left;
            }}
            .dataframe tbody td:nth-child(2){{
                text-align:center;
            }}
            .dataframe thead th,td {{
                padding: 7px
            }}
            .dataframe tbody td{{
                font-family: "Source Sans Pro",sans-serif;
                color: #000000;
                text-align: left;
                padding: 7px;
            }}
            .dataframe tbody tr {{
                border-bottom: 1px solid #dddddd;
                text-align: left;
            }}
            .dataframe tbody tr:nth-of-type(even) {{
                background-color: #f3f3f3;
            }}
            .h1 {{
                font-family: sans-serif;
            }}
        </style>
        <center>
        <table>
        <tr>
        <td>
        {df.to_html(index = False,)}
        </td>
        <td>
        {draw_smiles(smiles, elem_id = "id_2d", scale=0, height = 200)}
        </td>
        </table>
        </center>
    """
    st.components.v1.html(html, height = 1000)

    # html = f"""
       
    #     """
    # st.components.v1.html(html, height = 700)
