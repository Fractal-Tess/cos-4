{
  description = "A development environment for Node.js and Python";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    systems.url = "github:nix-systems/default";
  };

  outputs =
    { nixpkgs, systems, ... }:
    let
      eachSystem =
        f: nixpkgs.lib.genAttrs (import systems) (system: f (import nixpkgs { inherit system; }));
    in
    {
      devShells = eachSystem (pkgs: {
        default = pkgs.mkShell {
          shellHook = ''
            echo "
            uv - $(${pkgs.uv}/bin/uv --version)
            " | ${pkgs.lolcat}/bin/lolcat
          '';
          LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.glibc}/lib:$LD_LIBRARY_PATH";
          buildInputs = with pkgs; [
            stdenv.cc.cc.lib
            glibc
          ];
          packages = with pkgs; [
            # Node.js runtime
            bun

            uv

            # C++ standard library and runtime dependencies
            gcc
            glibc
            stdenv.cc.cc.lib
          ];
        };
      });
    };
}
