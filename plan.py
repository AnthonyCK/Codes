import numpy as np
import copy

def value_iteration(states, v0, tpd, params, unit, epsilon, lbd):
    v = copy.deepcopy(v0)
    vtm1 = copy.deepcopy(v0)
    action_dict = dict()
    for iteration in range(100):
        print(iteration)
        for state in states:
            actions = all_actions(state, params)
            candidate = np.zeros(len(actions))
            for index, action in enumerate(actions):
                candidate[index] = cost(state, action, params, unit)
                next_states = next_states_gen(state, action, params)
                for nextstate in next_states:
                    ind = np.where(states == nextstate)[0][0]
                    candidate[index] += lbd*full_transition_probability(tpd, state, nextstate, params)*v[ind]
            if len(candidate)>0:
                ind = np.where(states==state)[0][0]
                v[ind] = np.min(candidate)
                action_dict[state] = actions[np.argmin(candidate)]
            else:
                ind = np.where(states == state)[0][0]
                print("state {}".format(state))
                print(ind)
                raise Exception("Error!")

        nrm = np.max(np.abs(v-vtm1))
        if  nrm<= epsilon*(1-lbd)/(2*lbd):
            break
        else:
            vtm1 = copy.deepcopy(v)
    return v, action_dict

def full_transition_probability(tp_d, st, stp1, setting_params):
    dtm1 = st % (setting_params['max D'])
    dt = stp1 % (setting_params['max D'])
    bt = (st // setting_params['max D']) % setting_params['max s']
    btp1 = (stp1 // setting_params['max D']) % setting_params['max s']
    invt = st // (setting_params['max D']*setting_params['max b'])
    invtp1 = stp1 // (setting_params['max D'] * setting_params['max b'])
    if bt*invt == 0 and btp1*invtp1 == 0:
        #dt = st["s"] - stp1["s"] -st["b"] + stp1["b"] + at
        return  tp_d[dtm1,dt]
    else:
        return  0

def cost(st, at, params, unit):
    c = 0
    bt = (st // params['max D']) % params['max s']
    invt = st // (params['max D'] * params['max b'])
    if invt + at >0:
        c += (params["fixed holding cost"] + params["variable holding cost"]*unit*(at+invt))
    if at > 0:
        c += (params["fixed production cost"] + params["variable production cost"]*unit*at)
    if bt > 0:
        c += (params["fixed backlog cost"] + params["variable backlog cost"] * unit * bt)
    return c

def all_actions(state, params):
    invt = state // (params['max D'] * params['max b'])
    maxproduce = params["max s"] - invt
    bt = (state // params['max D']) % params['max s']
    minproduce = max(bt+params['max D']-params['max b'], 0)
    if maxproduce < minproduce:
        print("backlog {}".format(bt))
        print("inventory {}".format(invt))
        print("maxproduce {}".format(maxproduce))
        print("minproduce {}".format(minproduce))
    return list(range(minproduce, maxproduce+1))

def next_states_gen(state, action, params):
    res = []
    bt = (state // params['max D']) % params['max s']
    invt = state % (params['max D'] * params['max b'])
    for dt in range(params['max D']):
        invtp1 = max(invt - bt + action - dt, 0)
        if invtp1 >= params['max s']:
            continue
        btp1 = max(bt - invt + dt - action, 0)
        if btp1 >= params['max b']:
            continue
        res.append(dt+btp1*params['max D']+invtp1*params['max D']*params['max s'])

    return res

def print_actions(ad, params):
    for state in ad.keys():
        invt = state // (params['max D'] * params['max b'])
        bt = (state // params['max D']) % params['max s']
        dt = state % params['max b']
        print("Inventory level {}, backlog {}, demand in last period is {}, action is to produce {}".format(invt, bt, dt, ad[state]))

if __name__ == "__main__":
     doc1 = open("parameters/parameters.txt","r")
     params = doc1.read()
     params = eval(params)
     doc1.close()
     doc2 = open("parameters/settings.txt", "r")
     params2 = doc2.read()
     params2 = eval(params2)
     doc2.close()
     unit = params2["M"]/params2["max s"]
     all_states = []
     for dt in range(params2["max D"]):
         for inv in range(params2["max s"]):
             for b in range(params2["max b"]):
                 if inv*b == 0:
                     one_state = inv*params2["max b"]*params2["max D"]+b*params2["max D"]+ dt
                     all_states.append(one_state)
     all_states = np.sort(np.array(all_states))
     v_dict = all_states
     v0 = np.zeros(len(all_states))
     tpd = np.abs(np.random.normal(size=(len(all_states),len(all_states))))
     tot = tpd.sum(axis=1)
     tpd = tpd/tot[:,None]
     v, ad= value_iteration(all_states, v0, tpd, dict(params2,**params), unit, 0.01, 0.9)
     print(v)
     print_actions(ad, params2)
     #print(ad)
     '''
     print(params)
     print(dict(params2,**params))
     print(all_states)
'''

