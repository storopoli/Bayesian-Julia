# This file was generated, do not modify it. # hide
using Downloads
using DataFrames
using CSV
using Chain
using Dates

url = "https://data.brasil.io/dataset/covid19/caso_full.csv.gz"
file = Downloads.download(url)
df = DataFrame(CSV.File(file))
br = @chain df begin
    filter(
        [:date, :city] =>
            (date, city) ->
                date < Dates.Date("2021-01-01") &&
                    date > Dates.Date("2020-04-01") &&
                    ismissing(city),
        _,
    )
    groupby(:date)
    combine(
        [
            :estimated_population_2019,
            :last_available_confirmed_per_100k_inhabitants,
            :last_available_deaths,
            :new_confirmed,
            :new_deaths,
        ] .=>
            sum .=> [
                :estimated_population_2019,
                :last_available_confirmed_per_100k_inhabitants,
                :last_available_deaths,
                :new_confirmed,
                :new_deaths,
            ],
    )
end;