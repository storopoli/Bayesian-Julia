# This file was generated, do not modify it. # hide
using DataFrames
using CSV
using HTTP

url = "https://raw.githubusercontent.com/storopoli/Bayesian-Julia/master/datasets/duncan.csv"
duncan = CSV.read(HTTP.get(url).body, DataFrame)
describe(duncan)