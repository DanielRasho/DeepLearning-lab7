{
  description = "A basic multiplatform flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    # System types to support.
    supportedSystems = ["x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin"];

    # Helper function to generate an attrset '{ x86_64-linux = f "x86_64-linux"; ... }'.
    forAllSystems = nixpkgs.lib.genAttrs supportedSystems;

    # Nixpkgs instantiated for supported system types.
    nixpkgsFor = forAllSystems (system: import nixpkgs {inherit system;});
  in {
    devShells = forAllSystems (system: let
      pkgs = nixpkgsFor.${system};
      python = pkgs.python3.withPackages (p: [
        p.jupyterlab
        p.jupyterlab-lsp
        p.jupyterlab-execute-time
        p.matplotlib
        p.numpy
        p.scipy
        p.scikit-image
        p.torch
        p.torchvision
        p.torchsummary
        p.pandas
        p.datasets
        p.nltk
      ]);
    in {
      default = pkgs.mkShell {
        packages = [
          pkgs.black
          python
          pkgs.jq
        ];
      };
    });
  };
}
