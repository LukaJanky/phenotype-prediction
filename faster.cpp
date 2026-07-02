#include <iostream>
#include <fstream>
#include <vector>
#include <bits/stdc++.h>

using namespace std;

int main() {
    vector<int> lab;
    vector<int> cancer;
    vector<int> no_cancer;

    ifstream fp("E-MTAB-3732.sdrf.txt");
    if (fp.is_open()) {
    string line;
    string cuv = "normal";
    while (getline(fp, line)) {
        size_t found = line.find(cuv);
        if (found != string::npos)
            lab.push_back(0);
        else
            lab.push_back(1); 
        }
        fp.close();
    }
    lab.erase(lab.begin());
    
    for (int i = 0; i < lab.size(); i++) {
        if(lab[i] == 1){
            cancer.push_back(i);
        }
        else{
            no_cancer.push_back(i);
        }
    }
    vector<int> cancer_2(cancer.begin(), cancer.begin() + 6500);
    vector<int> no_cancer_2(no_cancer.begin(), no_cancer.begin() + 3500);

    ifstream fi("processedMatrix.Aurora.july2015.txt");
    if (fi.is_open()) {
        string linie;
        string s;
        vector <string> vec1;
        vector <string> vec2;
        //while (getline(fi, linie)) {
            //cout << linie <<endl;   
        //}
        while (getline(fi, linie)) {
            //getline(fi, linie);
            stringstream ss(linie);
            vector<string> v;
        
            while (getline(ss, s, '\t')) {
                v.push_back(s);
            }
            v.erase(v.begin());
            for (int idx = 0; idx < v.size(); idx++) {
                string ele = v[idx];
                if (find(cancer_2.begin(), cancer_2.end(), idx) != cancer_2.end()) {
                    vec1.push_back(ele);
                }
            }
            for (int id = 0; id < v.size(); id++) {
                string elem = v[id];
                if (std::find(no_cancer_2.begin(), no_cancer_2.end(), id) != no_cancer_2.end()) {
                    vec2.push_back(elem);
                }
            }
            vec1.insert(vec1.end (), vec2.begin(), vec2.end ());
            //cout << vec1.size() << endl;
            ofstream myfile("proces_extracted.txt", std::ios::app);
            if(myfile.is_open())
            {
                ostream_iterator<string> output_iterator(myfile, "\t");
                copy(begin(vec1), end(vec1), output_iterator);
                myfile << endl;
            }
            vec1.clear();
            vec2.clear();
            myfile.close();
        }
        fi.close();
    }
    return 0;
}