# This file was generated, do not modify it. # hide
abstract type Pet end
struct Dog <: Pet
    name::String
end
struct Cat <: Pet
    name::String
end

function encounter(a::Pet, b::Pet)
    verb = meets(a, b)
    return println("$(a.name) meets $(b.name) and $verb")
end

meets(a::Dog, b::Dog) = "sniffs";
meets(a::Dog, b::Cat) = "chases";
meets(a::Cat, b::Dog) = "hisses";
meets(a::Cat, b::Cat) = "slinks";