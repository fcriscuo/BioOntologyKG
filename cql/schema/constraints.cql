// Create constraints and indexes for genomic data

// Gene constraints
CREATE CONSTRAINT gene_id_unique IF NOT EXISTS FOR (g:Gene) REQUIRE g.id IS UNIQUE;
CREATE INDEX gene_symbol_index IF NOT EXISTS FOR (g:Gene) ON (g.symbol);

// Variant constraints
CREATE CONSTRAINT variant_id_unique IF NOT EXISTS FOR (v:Variant) REQUIRE v.id IS UNIQUE;
CREATE INDEX variant_position_index IF NOT EXISTS FOR (v:Variant) ON (v.chromosome, v.position);

// Sample constraints
CREATE CONSTRAINT sample_id_unique IF NOT EXISTS FOR (s:Sample) REQUIRE s.id IS UNIQUE;
