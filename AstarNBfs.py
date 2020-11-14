import numpy as np
import pandas as pd

nodefile = pd.read_csv("C:\\Users\\70743\\Desktop\\node.txt",header = None)
edgefile = pd.read_csv("C:\\Users\\70743\\Desktop\\edge.txt",header = None)
node = (np.array(nodefile))[:,1:4].astype(np.float32)
edge = np.round((np.array(edgefile))[:,1:4].astype(np.float32)).astype('int32')

def manhattan(m, n):
    return 1000 * np.sum(np.abs(m - n), 1)

def runAstar(start_node,end_node):
    openlist = np.array([start_node])
    openlistf = np.array([-1])
    olistG = np.array([0])
    aindex = np.round(node[:,0]).astype('int32') == start_node
    bindex = np.round(node[:,0]).astype('int32') == end_node
    olistH = manhattan(node[aindex, 1:3],node[bindex, 1:3])
    closelist = np.array([])
    closelistf = np.array([])
    while True:
        minoli = np.argmin(olistG+olistH)
        minol = np.array([openlist[minoli]])
        closelist = (np.append(closelist,minol)).astype('int32')
        closelistf = np.append(closelistf,np.array([openlistf[minoli]]))
        openlist = np.delete(openlist,minoli)
        openlistf = np.delete(openlistf,minoli)
        fatherG = olistG[minoli]
        olistG = np.delete(olistG,minoli)
        olistH = np.delete(olistH,minoli)
        newindex = np.where(edge[:,0] == minol)[0]
        newol = edge[newindex,1]
        tmp = (np.intersect1d(newol,closelist)).astype('int32')
        inter = np.array([])
        for val in tmp:
            inter = np.append(inter,np.where(newol == val)[0])
        inter = inter.astype('int32')
        newindex = np.delete(newindex,inter)
        newol = np.delete(newol,inter)
        openlist = np.append(openlist,newol)
        openlistf = np.append(openlistf,minol*np.ones(len(newol)))
        olistG = np.append(olistG,edge[newindex,2]+fatherG)
        aindex = newol.astype('int32')-1
        bindex = (end_node*np.ones(len(aindex))).astype('int32')-1
        olistH = np.append(olistH, manhattan(node[aindex,1:3],node[bindex,1:3]))
        if closelist[-1] == end_node:
            break
    closelistf = closelistf.astype('int32')
    clfi = closelistf[-1]
    path = np.array(clfi)
    while clfi != start_node:
        a = closelist == clfi
        clfi = closelistf[a][0]
        path = np.append(path,clfi)
    path = np.append(end_node,path)
    return np.append(fatherG,path)

result = runAstar(311,1)
print(311,1,result[0])
print(result[1:])

def runbfs(boom):
    aclub = np.array([boom])
    ad = np.array([0])
    bclub = np.array([])
    bd = np.array([])
    cclub = np.array([])
    cd = np.array([])
    while True:
        tmp = aclub[0]
        tmpd = ad[0]
        aclub = np.delete(aclub,0)
        ad = np.delete(ad,0)
        bclub = np.append(bclub,tmp)
        bd = np.append(bd,tmpd)
        newnode = edge[edge[:,0] == tmp,1]
        newd = edge[edge[:,0] == tmp,2]+tmpd
        intera = np.intersect1d(newnode,aclub)
        interb = np.intersect1d(newnode,bclub)
        interc = np.intersect1d(newnode,cclub)
        club = np.append(intera,interb)
        club = np.append(club,interc)
        inter = np.array([])
        for val in club:
            inter = np.append(inter,np.where(newnode == val)[0])
        inter = inter.astype('int32')
        newnode = np.delete(newnode,inter)
        newd = np.delete(newd,inter)
        for val in range(len(newnode)):
            if newd[val] < 100000:
                aclub = np.append(aclub,newnode[val])
                ad = np.append(ad,newd[val])
            else:
                cclub = np.append(cclub,newnode[val])
                cd = np.append(cd,newd[val])
        if len(aclub) == 0:
            break
    return bclub

print(runbfs(311))
