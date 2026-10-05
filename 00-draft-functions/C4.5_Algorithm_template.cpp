#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <cmath>
#include <memory>
#include <algorithm>

using namespace std;

// ---------- Data Structures ----------

struct Instance {
    vector<string> attrs;   // attribute values
    string label;           // class label
};

struct Node {
    bool isLeaf = false;
    string prediction;                 // for leaf
    int attrIndex = -1;                // for internal node
    map<string, shared_ptr<Node>> children;
};

// ---------- Utility Functions ----------

// Count label frequencies in a subset
map<string, int> countLabels(const vector<Instance>& data) {
    map<string, int> counts;
    for (auto& inst : data) counts[inst.label]++;
    return counts;
}

// Entropy of a label distribution
double entropy(const vector<Instance>& data) {
    auto counts = countLabels(data);
    double e = 0.0;
    for (auto& [label, c] : counts) {
        double p = (double)c / data.size();
        e -= p * log2(p);
    }
    return e;
}

// Information Gain for a discrete attribute
double infoGain(const vector<Instance>& data, int attrIdx) {
    double baseEntropy = entropy(data);
    map<string, vector<Instance>> partitions;

    for (auto& inst : data)
        partitions[inst.attrs[attrIdx]].push_back(inst);

    double weightedEntropy = 0.0;
    for (auto& [val, subset] : partitions) {
        double w = (double)subset.size() / data.size();
        weightedEntropy += w * entropy(subset);
    }
    return baseEntropy - weightedEntropy;
}

// Split Information (intrinsic value) for gain ratio
double splitInfo(const vector<Instance>& data, int attrIdx) {
    map<string, int> counts;
    for (auto& inst : data) counts[inst.attrs[attrIdx]]++;
    double si = 0.0;
    for (auto& [val, c] : counts) {
        double p = (double)c / data.size();
        si -= p * log2(p);
    }
    return si;
}

// Gain Ratio = InfoGain / SplitInfo
double gainRatio(const vector<Instance>& data, int attrIdx) {
    double ig = infoGain(data, attrIdx);
    double si = splitInfo(data, attrIdx);
    if (si == 0.0) return 0.0; // avoid divide by zero
    return ig / si;
}

// Majority label
string majorityLabel(const vector<Instance>& data) {
    auto counts = countLabels(data);
    string best;
    int bestCount = -1;
    for (auto& [label, c] : counts) {
        if (c > bestCount) { bestCount = c; best = label; }
    }
    return best;
}

// ---------- C4.5 Tree Building ----------

shared_ptr<Node> buildTree(
    const vector<Instance>& data,
    const set<int>& remainingAttrs,
    const vector<string>& attrNames)
{
    auto node = make_shared<Node>();

    // Case 1: All instances have the same label -> leaf
    auto counts = countLabels(data);
    if (counts.size() == 1) {
        node->isLeaf = true;
        node->prediction = data[0].label;
        return node;
    }

    // Case 2: No attributes left -> leaf with majority label
    if (remainingAttrs.empty()) {
        node->isLeaf = true;
        node->prediction = majorityLabel(data);
        return node;
    }

    // Pick the attribute with the best gain ratio
    int bestAttr = -1;
    double bestRatio = -1.0;
    for (int idx : remainingAttrs) {
        double gr = gainRatio(data, idx);
        if (gr > bestRatio) {
            bestRatio = gr;
            bestAttr = idx;
        }
    }

    // If no useful split, return majority
    if (bestAttr == -1 || bestRatio <= 0.0) {
        node->isLeaf = true;
        node->prediction = majorityLabel(data);
        return node;
    }

    node->attrIndex = bestAttr;
    cout << "Splitting on: " << attrNames[bestAttr]
         << " (gain ratio = " << bestRatio << ")\n";

    // Partition data by attribute values
    map<string, vector<Instance>> partitions;
    for (auto& inst : data)
        partitions[inst.attrs[bestAttr]].push_back(inst);

    // Build subtree for each branch
    set<int> newRemaining = remainingAttrs;
    newRemaining.erase(bestAttr);

    for (auto& [val, subset] : partitions) {
        cout << "  Branch " << attrNames[bestAttr]
             << " = " << val << " (" << subset.size() << " instances)\n";
        node->children[val] = buildTree(subset, newRemaining, attrNames);
    }

    return node;
}

// ---------- Prediction ----------

string predict(const shared_ptr<Node>& node,
               const vector<string>& attrNames,
               const map<string, string>& query)
{
    if (node->isLeaf) return node->prediction;

    string attrName = attrNames[node->attrIndex];
    string val = query.at(attrName);

    auto it = node->children.find(val);
    if (it == node->children.end()) {
        // Unknown value at prediction time: fall back (simple case)
        return "Unknown";
    }
    return predict(it->second, attrNames, query);
}

// ---------- Pretty Print Tree ----------

void printTree(const shared_ptr<Node>& node,
               const vector<string>& attrNames,
               const string& indent = "")
{
    if (node->isLeaf) {
        cout << indent << "-> " << node->prediction << "\n";
        return;
    }
    cout << indent << "[" << attrNames[node->attrIndex] << "]\n";
    for (auto& [val, child] : node->children) {
        cout << indent << "  " << val << ":\n";
        printTree(child, attrNames, indent + "    ");
    }
}

// ---------- Main ----------

int main() {
    vector<string> attrNames = {"Outlook", "Temp", "Humidity", "Wind"};

    vector<Instance> data = {
        {{"Sunny","Hot","High","Weak"},       "No"},
        {{"Sunny","Hot","High","Strong"},     "No"},
        {{"Overcast","Hot","High","Weak"},    "Yes"},
        {{"Rain","Mild","High","Weak"},       "Yes"},
        {{"Rain","Cool","Normal","Weak"},     "Yes"},
        {{"Rain","Cool","Normal","Strong"},   "No"},
        {{"Overcast","Cool","Normal","Strong"},"Yes"},
        {{"Sunny","Mild","High","Weak"},      "No"},
        {{"Sunny","Cool","Normal","Weak"},    "Yes"},
        {{"Rain","Mild","Normal","Weak"},     "Yes"},
        {{"Sunny","Mild","Normal","Strong"},  "Yes"},
        {{"Overcast","Mild","High","Strong"}, "Yes"},
        {{"Overcast","Hot","Normal","Weak"},  "Yes"},
        {{"Rain","Mild","High","Strong"},     "No"}
    };

    // All attribute indices initially available
    set<int> remainingAttrs;
    for (int i = 0; i < (int)attrNames.size(); ++i)
        remainingAttrs.insert(i);

    cout << "=== Building C4.5 Tree ===\n";
    auto tree = buildTree(data, remainingAttrs, attrNames);

    cout << "\n=== Tree Structure ===\n";
    printTree(tree, attrNames);

    // Test prediction
    cout << "\n=== Predictions ===\n";
    vector<map<string,string>> queries = {
        {{"Outlook","Sunny"},{"Temp","Cool"},{"Humidity","High"},{"Wind","Strong"}},
        {{"Outlook","Overcast"},{"Temp","Mild"},{"Humidity","Normal"},{"Wind","Weak"}},
        {{"Outlook","Rain"},{"Temp","Mild"},{"Humidity","High"},{"Wind","Weak"}}
    };

    for (auto& q : queries) {
        cout << "Outlook=" << q["Outlook"]
             << " Temp="   << q["Temp"]
             << " Humidity=" << q["Humidity"]
             << " Wind="   << q["Wind"]
             << "  =>  " << predict(tree, attrNames, q) << "\n";
    }

    return 0;
}