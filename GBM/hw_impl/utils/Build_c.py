from Tree_Reader import tree_reader
import os

output_type="ap_fixed<16,4>"
feature_num=4
input_type=["ap_fixed<8,4>","ap_fixed<8,4>","ap_fixed<8,4>","ap_fixed<8,4>"]

def tab_string(num,string):
    str=""
    for i in range(num):
        str+="\t"
    str+=string
    return str

def Build_Entry(trees,path,head,entry,str):
    f=open(path+"/"+entry+".cpp",'a')
    f.write("#include \""+head+".h\"\n")
    f.write("#include <ap_fixed.h>\n\n")
    f.write(output_type+" "+entry+"("+str+"){\n")
    f.write("\t"+output_type+" ans=0;\n")
    tmp_str=""
    for i in range(feature_num):
        if i!=0: tmp_str+=", "
        tmp_str+="f{}".format(i)
    for i in range(len(trees)):
        f.write("\tans += tree_{:02}(".format(i)+tmp_str+");\n")
    f.write("\treturn ans\n;")
    f.write("}")
    f.close()

def Build_head(trees, path, head, entry, str):
    f=open(path+"/"+head+".h",'a')
    f.write("#include <ap_fixed.h>\n\n")
    f.write(output_type + " " + entry + "(" + str + ");\n")
    for i in range(len(trees)):
        f.write(output_type + " " + "tree_{:02}".format(i) + "("+str+");\n")
    f.close()

def traceback(tree, file, tab_num):
    if tree.leaf:
        file.write(tab_string(tab_num,"return {};\n".format(tree.w)))
    else:
        file.write(tab_string(tab_num,"if(f{}<{})\n".format(tree.f_id,tree.val)))
        traceback(tree.left,file,tab_num+1)
        file.write(tab_string(tab_num,"else\n"))
        traceback(tree.right,file,tab_num+1)

def Build_c(tree, path, filename, head,str):
    f = open(path + "/"+filename+".cpp", 'a')
    f.write("#include \""+head+".h\"\n")
    f.write("#include <ap_fixed.h>\n\n")
    f.write(output_type+" "+filename+"("+str+"){\n")
    traceback(tree,f,1)
    f.write("}")
    f.close()


def Builder(trees, path, head, entry):
    str = ""
    for i in range(feature_num):
        if i != 0:
            str += ", "
        str += input_type[i] + " f{}".format(i)
    Build_head(trees,path,head,entry,str)
    Build_Entry(trees,path,head,entry,str)
    for index, tree in enumerate(trees):
        Build_c(tree,path,"tree_{:02}".format(index),head,str)
        print(index)


if __name__ == '__main__':
    doc_path="trees_c"
    model_path= "file/model_iris.txt"
    head="xgb"
    entry="xgb"
    if os.path.exists(doc_path) == False:
        os.mkdir(doc_path)
    trees = tree_reader(model_path)
    Builder(trees, doc_path, head, entry)
