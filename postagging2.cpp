#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include "maxent.h"

using namespace std;

struct Token
{
  string str;
  string pos;
  string answer;
  Token(const string & s, const string & p,const string &a) : str(s), pos(p),answer(a) {}
};
std::vector<std::string> StringSplit(std::string sstr, const char* delim)
{
  std::vector<std::string> results;
  char *src = new char [sstr.length() + 1];
  strncpy(src,sstr.c_str(),sstr.length());
  src[sstr.length()] = 0;
  char *p = strtok(src,delim);
  if ( p!= NULL)
  {
    results.push_back(p);
  }
  while ( (p=strtok(NULL,delim)) != NULL )
  {
    results.push_back(p);
  }
  if (src != NULL )
  {
    delete [] src;
    src = NULL;
  }
  return results;
}

ME_Sample sample(const vector<Token> & vt, int i)
{
  ME_Sample sample;

  sample.label = vt[i].answer;

  const string & w0 = vt[i].str;
  const string wp1 = i > 0 ? vt[i - 1].str : "BOS";
  const string wp2 = i > 1 ? vt[i - 2].str : "BOS";
  const string wm1 = i < (int)vt.size() - 1 ? vt[i + 1].str : "EOS";
  const string wm2 = i < (int)vt.size() - 2 ? vt[i + 2].str : "EOS";

  sample.add_feature("W0_"  + w0);
  sample.add_feature("W-1_" + wm1);
  sample.add_feature("W+1_" + wp1);
  sample.add_feature("W-2_" + wm2);
  sample.add_feature("W+2_" + wp2);

  sample.add_feature("W-10_"  + wm1 + "_" + w0);
  sample.add_feature("W0+1_"  + w0  + "_" + wp1);

  const string & p0 = vt[i].pos;
  const string pp1 = i > 0 ? vt[i - 1].pos : "BOS";
  const string pp2 = i > 1 ? vt[i - 2].pos : "BOS";
  const string pm1 = i < (int)vt.size() - 1 ? vt[i + 1].pos : "EOS";
  const string pm2 = i < (int)vt.size() - 2 ? vt[i + 2].pos : "EOS";

  sample.add_feature("p0_"  + p0);
  sample.add_feature("p-1_" + pm1);
  sample.add_feature("p+1_" + pp1);
  sample.add_feature("p-2_" + pm2);
  sample.add_feature("p+2_" + pp2);

  sample.add_feature("p-10_"  + pm1 + "_" + p0);
  sample.add_feature("p0+1_"  + p0  + "_" + pp1);
  sample.add_feature("p+1+2_"  + pp1  + "_" + pp2);
  sample.add_feature("p-1-2_"  + pm1  + "_" + pm2);
  sample.add_feature("p0-1-2_" + p0 + pm1  + "_" + pm2);
  sample.add_feature("p0+1+2_" + p0 + pp1  + "_" + pp2);
  sample.add_feature("p-1+0+1_" + pm1 + p0  + "_" + pp1);
/*
  char buf[1000];
  for (unsigned int j = 1; j <= 10; j++) {
    if (w0.size() >= j) {
      sprintf(buf, "SUF_%s", w0.substr(w0.size() - j).c_str());
      sample.add_feature(buf);
    }
    if (w0.size() >= j) {
      sprintf(buf, "PRE_%s", w0.substr(0, j).c_str());
      sample.add_feature(buf);
    }
  }
  */
  return sample;
}

vector<Token> read_line(const string & line) 
{
  std::vector<Token>vs;
  std::vector<std::string>words = StringSplit(line,"#");
  for (int i = 0; i < words.size();i++)
  {
    std::vector<std::string> elems = StringSplit(words[i],",");
    vs.push_back(Token(elems[0],elems[1],elems[2]));
  }
    
  return vs;
}

void train(ME_Model & model, const string & filename)
{
  ifstream ifile(filename.c_str());
  
  if (!ifile) {
    cerr << "error: cannot open " << filename << endl; 
    exit(1); 
  }

  string line;
  int n = 0;
  while (getline(ifile, line)) {
    vector<Token> vs = read_line(line);
    for (int j = 0; j < (int)vs.size(); j++) {
      ME_Sample mes = sample(vs, j);
      model.add_training_sample(mes);
    }
    if (n++ > 10000) break;
  }    

  model.use_l1_regularizer(1.0);
  //  model.use_l2_regularizer(1.0);
  //  model.use_SGD();
  model.set_heldout(100);
  model.train();
  model.save_to_file("model");
}

void test(const ME_Model & model, const string & filename,const string & ofilename) 
{
  ifstream ifile(filename.c_str());
  std::ofstream fout(ofilename.c_str());
  
  if (!ifile) {
    cerr << "error: cannot open " << filename << endl; 
    exit(1); 
  }

  int num_correct = 0;
  int num_tokens = 0;
  string line;
  while (getline(ifile, line)) {
    std::vector<std::string> words =StringSplit(line,"#");
    vector<Token> vs = read_line(line);
    for (int j = 0; j < (int)vs.size(); j++) {
      ME_Sample mes = sample(vs, j);
      model.classify(mes);
      std::vector<std::string> elem = StringSplit(words[j],",");
      fout<<elem[0]<<"\t"<<elem[1]<<"\t"<<elem[2]<<"\t"<<mes.label<<std::endl;
      if (mes.label == vs[j].pos) num_correct++;
      num_tokens++;
    }
    fout<<std::endl;
  }    
  cout << "accuracy = " << num_correct << " / " << num_tokens << " = " 
       << (double)num_correct / num_tokens << endl;
  ifile.close();
  fout.close();
}

int main()
{
  ME_Model m;
  train(m, "./sample_data/trainset.maxent.txt");
  test(m,  "./sample_data/th2.maxent.txt","./sample_data/result.maxent.txt");
}
